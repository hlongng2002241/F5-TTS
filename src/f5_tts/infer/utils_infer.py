# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format
import os
import sys
from concurrent.futures import ThreadPoolExecutor


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../third_party/BigVGAN/")

import hashlib
import re
import tempfile
from importlib.resources import files

import matplotlib


matplotlib.use("Agg")

import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
import tqdm
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos

from f5_tts.model import CFM
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer, list_str_to_idx


_ref_audio_cache = {}
_ref_text_cache = {}

device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu" if torch.xpu.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

tempfile_kwargs = {"delete_on_close": False} if sys.version_info >= (3, 12) else {"delete": False}

# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

# -----------------------------------------


# chunk text into smaller pieces


def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# load vocoder
def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device, hf_cache_dir=None):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print("You need to follow the README to init submodule and change the BigVGAN source code.")
        if is_local:
            # download generator from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            vocoder = bigvgan.BigVGAN.from_pretrained(
                "nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False, cache_dir=hf_cache_dir
            )

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder


# load asr pipeline

asr_pipe = None


def initialize_asr_pipeline(device: str = device, dtype=None):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )


# transcribe


def transcribe(ref_audio, language=None):
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline(device=device)
    return asr_pipe(
        ref_audio,
        chunk_length_s=30,
        batch_size=128,
        generate_kwargs={"task": "transcribe", "language": language} if language else {"task": "transcribe"},
        return_timestamps=False,
    )["text"].strip()


# load model checkpoint for inference


def load_checkpoint(model, ckpt_path, device: str, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v for k, v in checkpoint["ema_model_state_dict"].items() if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device)


# load model for inference


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
):
    if vocab_file == "":
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("token : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    return model


def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio


# preprocess reference audio and text


def preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=print):
    show_info("Converting audio...")

    # Compute a hash of the reference audio file
    with open(ref_audio_orig, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    global _ref_audio_cache

    if audio_hash in _ref_audio_cache:
        show_info("Using cached preprocessed reference audio...")
        ref_audio = _ref_audio_cache[audio_hash]

    else:  # first pass, do preprocess
        with tempfile.NamedTemporaryFile(suffix=".wav", **tempfile_kwargs) as f:
            temp_path = f.name

        aseg = AudioSegment.from_file(ref_audio_orig)

        # 1. try to find long silence for clipping
        non_silent_segs = silence.split_on_silence(
            aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                show_info("Audio is over 12s, clipping short. (1)")
                break
            non_silent_wave += non_silent_seg

        # 2. try to find short silence for clipping if 1. failed
        if len(non_silent_wave) > 12000:
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                    show_info("Audio is over 12s, clipping short. (2)")
                    break
                non_silent_wave += non_silent_seg

        aseg = non_silent_wave

        # 3. if no proper silence found for clipping
        if len(aseg) > 12000:
            aseg = aseg[:12000]
            show_info("Audio is over 12s, clipping short. (3)")

        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(temp_path, format="wav")
        ref_audio = temp_path

        # Cache the processed reference audio
        _ref_audio_cache[audio_hash] = ref_audio

    if not ref_text.strip():
        global _ref_text_cache
        if audio_hash in _ref_text_cache:
            # Use cached asr transcription
            show_info("Using cached reference text...")
            ref_text = _ref_text_cache[audio_hash]
        else:
            show_info("No reference text provided, transcribing reference audio...")
            ref_text = transcribe(ref_audio)
            # Cache the transcribed text (not caching custom ref_text, enabling users to do manual tweak)
            _ref_text_cache[audio_hash] = ref_text
    else:
        show_info("Using custom reference text...")

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    print("\nref_text  ", ref_text)

    return ref_audio, ref_text


# infer process: chunk text -> infer batches [i.e. infer_batch_process()]


def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
):
    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr) * speed)
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    for i, gen_text in enumerate(gen_text_batches):
        print(f"gen_text {i}", gen_text)
    print("\n")

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    return next(
        infer_batch_process(
            (audio, sr),
            ref_text,
            gen_text_batches,
            model_obj,
            vocoder,
            mel_spec_type=mel_spec_type,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
        )
    )


# infer batches


def infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    mel_spec_type="vocos",
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
    streaming=False,
    chunk_size=2048,
):
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    def process_batch(gen_text):
        local_speed = speed
        if len(gen_text.encode("utf-8")) < 10:
            local_speed = 0.3

        # Prepare the text
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        ref_audio_len = audio.shape[-1] // hop_length
        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            # Calculate duration
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)

        # inference
        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )
            del _

            generated = generated.to(torch.float32)  # generated mel spectrogram
            generated = generated[:, ref_audio_len:, :]
            generated = generated.permute(0, 2, 1)
            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated)
            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()

            if streaming:
                for j in range(0, len(generated_wave), chunk_size):
                    yield generated_wave[j : j + chunk_size], target_sample_rate
            else:
                generated_cpu = generated[0].cpu().numpy()
                del generated
                yield generated_wave, generated_cpu

    if streaming:
        for gen_text in progress.tqdm(gen_text_batches) if progress is not None else gen_text_batches:
            for chunk in process_batch(gen_text):
                yield chunk
    else:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_batch, gen_text) for gen_text in gen_text_batches]
            for future in progress.tqdm(futures) if progress is not None else futures:
                result = future.result()
                if result:
                    generated_wave, generated_mel_spec = next(result)
                    generated_waves.append(generated_wave)
                    spectrograms.append(generated_mel_spec)

        if generated_waves:
            if cross_fade_duration <= 0:
                # Simply concatenate
                final_wave = np.concatenate(generated_waves)
            else:
                # Combine all generated waves with cross-fading
                final_wave = generated_waves[0]
                for i in range(1, len(generated_waves)):
                    prev_wave = final_wave
                    next_wave = generated_waves[i]

                    # Calculate cross-fade samples, ensuring it does not exceed wave lengths
                    cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                    cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                    if cross_fade_samples <= 0:
                        # No overlap possible, concatenate
                        final_wave = np.concatenate([prev_wave, next_wave])
                        continue

                    # Overlapping parts
                    prev_overlap = prev_wave[-cross_fade_samples:]
                    next_overlap = next_wave[:cross_fade_samples]

                    # Fade out and fade in
                    fade_out = np.linspace(1, 0, cross_fade_samples)
                    fade_in = np.linspace(0, 1, cross_fade_samples)

                    # Cross-faded overlap
                    cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                    # Combine
                    new_wave = np.concatenate(
                        [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                    )

                    final_wave = new_wave

            # Create a combined spectrogram
            combined_spectrogram = np.concatenate(spectrograms, axis=1)

            yield final_wave, target_sample_rate, combined_spectrogram

        else:
            yield None, target_sample_rate, None


# remove silence from generated wav


def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10)
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")


# save spectrogram


def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()


def pad_1d(tensors: list[torch.Tensor], pad_value):
    lens = [int(t.size(0)) for t in tensors]
    max_len = max(lens)
    padded = torch.full((len(tensors), max_len), pad_value, device=tensors[0].device, dtype=tensors[0].dtype)
    for index, tensor in enumerate(tensors):
        padded[index, : tensor.shape[0]] = tensor
    return padded, lens


def pad_2d(tensors: list[torch.Tensor], pad_value):
    lens = [int(t.size(0)) for t in tensors]
    assert min(lens) == max(lens)

    lens = [int(t.size(1)) for t in tensors]
    max_len = max(lens)

    first = tensors[0]
    padded = torch.full((len(tensors), first.size(0), max_len), pad_value, device=first.device, dtype=first.dtype)

    for index, tensor in enumerate(tensors):
        padded[index, :, : tensor.size(1)] = tensor
    return padded, lens


def detect_bigvgan_mel_artifact_frames(mel_spec, n_frames=2, threshold_drop=-0.5):
    """
    Detect artifact frames at the end of mel spectrogram based on energy gradient analysis.

    For BigVGAN mel spectrograms (log-domain: torch.log(torch.clamp(mel, min=1e-5))).

    Key concept:
    - Natural speech endings have smooth energy decay (gradual gradient)
    - Artifact frames show sudden energy drops (steep negative gradient)

    Args:
        mel_spec: Mel spectrogram tensor/array of shape [mel_channels, time_frames]
                  Values are in log-domain (from BigVGAN mel extraction)
        n_frames: Number of last frames to evaluate (default: 2)
        threshold_drop: Gradient threshold for sudden drop detection (default: -0.5)
                        -0.5 = 50% energy drop indicates artifact
                        More negative = more sensitive (e.g., -0.3 detects 30% drops)

    Returns:
        int: Number of frames to truncate (0 to n_frames)
    """
    if n_frames == 0:
        return 0

    # Handle both torch tensors and numpy arrays
    if isinstance(mel_spec, torch.Tensor):
        mel_spec_np = mel_spec.cpu().numpy()
    else:
        mel_spec_np = mel_spec

    if mel_spec_np.shape[-1] < n_frames + 1:
        # Not enough frames to compute gradients, return default
        return 1

    # Convert from log-domain to linear-domain for energy calculation
    # BigVGAN mel: torch.log(torch.clamp(mel, min=1e-5))
    # Reverse: exp(log_mel) = linear_mel
    mel_linear = np.exp(mel_spec_np)

    # Calculate energy from linear-domain mel spectrogram
    frame_energy = np.mean(mel_linear, axis=0)  # [time_frames]

    # Compute energy gradients (percentage change between consecutive frames)
    # gradient[i] = (energy[i+1] - energy[i]) / energy[i]
    energy_gradients = []
    for i in range(len(frame_energy) - 1):
        if frame_energy[i] > 1e-8:  # Avoid division by zero
            gradient = (frame_energy[i + 1] - frame_energy[i]) / frame_energy[i]
        else:
            gradient = 0.0
        energy_gradients.append(gradient)

    # Check last n_frames for sudden changes (spikes OR drops)
    # Work backwards: if sudden change detected, truncate from there
    truncate_count = 0
    threshold_spike = abs(threshold_drop)  # Positive threshold for spikes

    for i in range(n_frames - 1, -1, -1):  # Check from last frame backwards
        # Gradient index: gradient between frame[i-1] and frame[i]
        gradient_idx = -(n_frames - i) - 1  # -1, -2, ... for last frames

        if gradient_idx >= -len(energy_gradients):
            gradient = energy_gradients[gradient_idx]

            # Detect sudden energy change (spike OR drop)
            # Artifact can be: sudden drop (gradient < -0.5) OR sudden spike (gradient > +0.5)
            if gradient < threshold_drop or gradient > threshold_spike:
                truncate_count = n_frames - i
            else:
                # Found frame with normal gradient, stop truncating
                break

    return truncate_count


def detect_vocos_mel_artifact_frames(mel_spec, n_frames=2, threshold_drop=-0.5):
    """
    Detect artifact frames at the end of mel spectrogram based on energy gradient analysis.

    For Vocos mel spectrograms (log-domain: mel.clamp(min=1e-5).log()).

    Key concept:
    - Natural speech endings have smooth energy decay (gradual gradient)
    - Artifact frames show sudden energy changes (steep negative/positive gradient)

    Args:
        mel_spec: Mel spectrogram tensor/array of shape [mel_channels, time_frames]
                  Values are in log-domain (from Vocos mel extraction)
        n_frames: Number of last frames to evaluate (default: 2)
        threshold_drop: Gradient threshold for sudden change detection (default: -0.5)
                        -0.5 = 50% energy drop/spike indicates artifact
                        More negative = more sensitive (e.g., -0.3 detects 30% changes)

    Returns:
        int: Number of frames to truncate (0 to n_frames)
    """
    if n_frames == 0:
        return 0

    # Handle both torch tensors and numpy arrays
    if isinstance(mel_spec, torch.Tensor):
        mel_spec_np = mel_spec.cpu().numpy()
    else:
        mel_spec_np = mel_spec

    if mel_spec_np.shape[-1] < n_frames + 1:
        # Not enough frames to compute gradients, return default
        return 1

    # Convert from log-domain to linear-domain for energy calculation
    # Vocos mel: mel.clamp(min=1e-5).log()
    # Reverse: exp(log_mel) = linear_mel
    mel_linear = np.exp(mel_spec_np)

    # Calculate energy from linear-domain mel spectrogram
    frame_energy = np.mean(mel_linear, axis=0)  # [time_frames]

    # Compute energy gradients (percentage change between consecutive frames)
    # gradient[i] = (energy[i+1] - energy[i]) / energy[i]
    energy_gradients = []
    for i in range(len(frame_energy) - 1):
        if frame_energy[i] > 1e-8:  # Avoid division by zero
            gradient = (frame_energy[i + 1] - frame_energy[i]) / frame_energy[i]
        else:
            gradient = 0.0
        energy_gradients.append(gradient)

    # Check last n_frames for sudden changes (spikes OR drops)
    # Work backwards: if sudden change detected, truncate from there
    truncate_count = 0
    threshold_spike = abs(threshold_drop)  # Positive threshold for spikes

    for i in range(n_frames - 1, -1, -1):  # Check from last frame backwards
        # Gradient index: gradient between frame[i-1] and frame[i]
        gradient_idx = -(n_frames - i) - 1  # -1, -2, ... for last frames

        if gradient_idx >= -len(energy_gradients):
            gradient = energy_gradients[gradient_idx]

            # Detect sudden energy change (spike OR drop)
            # Artifact can be: sudden drop (gradient < -0.5) OR sudden spike (gradient > +0.5)
            if gradient < threshold_drop or gradient > threshold_spike:
                truncate_count = n_frames - i
            else:
                # Found frame with normal gradient, stop truncating
                break

    return truncate_count


def infer_batch_synthesized_on_left(
    ref_audios: str | list[str],
    ref_texts: str | list[str],
    gen_texts: str | list[str],
    model_obj: CFM,
    vocoder: Vocos,
    speed: float | list[float] = speed,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    mel_spec_type="vocos",
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    device=device,
    n_mel_frames_trunc: int = None,
    fix_duration=None,
):
    assert mel_spec_type in ["vocos", "bigvgan"]
    if n_mel_frames_trunc is None:
        n_mel_frames_trunc = 3 if mel_spec_type == "vocos" else 2

    if isinstance(ref_audios, str):
        ref_audios = [ref_audios]
    if isinstance(ref_texts, str):
        ref_texts = [ref_texts]
    if isinstance(gen_texts, str):
        gen_texts = [gen_texts]
    if isinstance(speed, float):
        speed = [speed for _ in range(len(ref_audios))]

    assert len(ref_audios) == len(ref_texts)
    assert len(ref_audios) == len(gen_texts)
    assert len(ref_audios) == len(speed)

    audio_list = []
    text_list = []
    durations = []
    rms_list = []
    ref_audio_len_list = []

    sample_lens = []

    for ref_audio, ref_text, gen_text, local_speed in zip(ref_audios, ref_texts, gen_texts, speed):
        audio, sr = torchaudio.load(ref_audio)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        rms = torch.sqrt(torch.mean(torch.square(audio))).item()
        if rms < target_rms:
            audio = audio * target_rms / rms
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            audio = resampler(audio)

        max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr) * local_speed)
        gen_chunks = chunk_text(gen_text, max_chars=max_chars)
        sample_lens.append(len(gen_chunks))

        for gen_chunk in gen_chunks:
            gen_chunk = " " + gen_chunk + " "

            # TODO. currently, use the same speed for all samples
            if len(gen_chunk.encode("utf-8")) < 10:
                local_speed = 0.3

            audio_list.append(audio)  # audio shape is [1, N]
            rms_list.append(rms)

            ref_audio_len = audio.shape[-1] // hop_length
            ref_audio_len_list.append(ref_audio_len)

            text_list.append(convert_char_to_pinyin([gen_chunk + ref_text])[0])

            ref_text_len = len(ref_text.encode("utf-8"))
            gen_chunk_len = len(gen_chunk.encode("utf-8"))

            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_chunk_len / local_speed)
            durations.append(duration)

    duration = torch.LongTensor(durations).to(device)

    # inference
    with torch.inference_mode():
        cond = [model_obj.mel_spec(audio)[0] for audio in audio_list]
        cond, cond_lens = pad_2d(cond, pad_value=0.0)
        cond = cond.to(device).permute(0, 2, 1)
        cond_lens = torch.LongTensor(cond_lens).to(device)

        generated, trajectory = model_obj.sample_left(
            cond=cond,
            text=text_list,
            duration=duration,
            lens=cond_lens,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
        )
        del trajectory

        # generated mel spectrogram
        generated = generated.to(torch.float32)
        generated = generated.permute(0, 2, 1)

        # Detect artifact frames for each generated mel spectrogram
        mel_list = []
        mel_lengths = []
        mel_frames_trunc_list = []

        for index in range(generated.size(0)):
            # Get the mel spectrogram for this sample (excluding reference audio part)
            dur = durations[index]
            ref_audio_len = ref_audio_len_list[index]
            mel = generated[index, :, : dur - ref_audio_len]

            if mel_spec_type == "vocos":
                n_trunc = detect_vocos_mel_artifact_frames(mel, n_frames=n_mel_frames_trunc, threshold_drop=-0.5)
            else:
                n_trunc = detect_bigvgan_mel_artifact_frames(mel, n_frames=n_mel_frames_trunc, threshold_drop=-0.5)

            mel_frames_trunc_list.append(n_trunc)
            mel = mel[:, : mel.size(1) - n_trunc]
            mel_list.append(mel)
            mel_lengths.append(mel.size(1))

        print("mel_trunc_list =", mel_frames_trunc_list)
        print("mel_lengths =", mel_lengths)

        mel_list, _ = pad_2d(mel_list, pad_value=0.0)
        if mel_spec_type == "vocos":
            generated_waves = vocoder.decode(mel_list).cpu().numpy()  # [B, N]
        else:
            generated_waves = vocoder(mel_list).cpu().numpy()  # [B, N]

        generated_audio_list = []
        for generated_wave, rms, mel_len in zip(generated_waves, rms_list, mel_lengths):
            generated_wave = generated_wave[: mel_len * hop_length]
            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms
            generated_audio_list.append(generated_wave)

    start = 0
    final_audio_list: list[np.ndarray] = []
    for size in sample_lens:
        audio_chunks = generated_audio_list[start : start + size]
        start += size

        if cross_fade_duration <= 0:
            final_audio = np.concatenate(audio_chunks)
        else:
            final_audio = audio_chunks[0]
            for i in range(1, len(audio_chunks)):
                prev_chunk = final_audio
                next_chunk = audio_chunks[i]

                cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                cross_fade_samples = min(cross_fade_samples, len(prev_chunk), len(next_chunk))

                if cross_fade_samples <= 0:
                    # No overlap possible, concatenate
                    final_audio = np.concatenate([prev_chunk, next_chunk])
                    continue

                # Overlapping parts
                prev_overlap = prev_chunk[-cross_fade_samples:]
                next_overlap = next_chunk[:cross_fade_samples]

                # Fade out and fade in
                fade_out = np.linspace(1, 0, cross_fade_samples)
                fade_in = np.linspace(0, 1, cross_fade_samples)

                # Cross-faded overlap
                cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                # Combine
                final_audio = np.concatenate(
                    [prev_chunk[:-cross_fade_samples], cross_faded_overlap, next_chunk[cross_fade_samples:]]
                )

        final_audio_list.append(final_audio)

    return final_audio_list, target_sample_rate
