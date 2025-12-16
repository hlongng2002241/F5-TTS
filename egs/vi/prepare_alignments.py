import os
import json
import string
import jsonlines
from tqdm import tqdm
from collections import Counter, defaultdict

import librosa
import numpy as np
import matplotlib.pyplot as plt
from lhotse import CutSet
from lhotse.cut.mono import MonoCut
from lhotse.audio.recording import Recording, AudioSource
from lhotse.supervision import SupervisionSegment, AlignmentItem

from f5_tts.model.utils import match_alignment
from f5_tts.model.utils import convert_char_to_pinyin
from quick_utils.common.mp_utils import execute_parallel


def prepare(ali_path: str, data_path: str, output_path: str):
    item_id_to_ali = {}
    with jsonlines.open(ali_path) as f:
        for item in tqdm(f):
            item_id_to_ali[item["id"]] = item

    missing = 0
    error = []
    sample_rate = 24000
    hop_length = 256
    gaps = []
    with jsonlines.open(output_path, "w") as f_out, jsonlines.open(data_path) as f_in:
        for item in tqdm(f_in):
            if item["id"] not in item_id_to_ali:
                missing += 1
                continue
            ali = item_id_to_ali[item["id"]]
            assert item["audio_path"] == ali["audio_filepath"]
            text = [t.lower() for t in item["text"]]
            try:
                item["mel_alignments"], _gaps = match_alignment(
                    text,
                    ali["alignments"],
                    item["duration"],
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                    expand_gap=True,
                )
                gaps += _gaps
                f_out.write(item)
            except Exception as e:
                error.append(e.args[0])

    print("gaps =", len(gaps))
    for g in gaps[:10]:
        print(g)

    print("missing =", missing)
    print("error =", Counter(error))


def worker(item: dict):
    text = item["text"]
    text = text.replace("<sp>", " ")
    text = " ".join(text.split())
    tokens = convert_char_to_pinyin([text])[0]
    audio_path = item["audio_filepath"]
    y, sr = librosa.load(audio_path, sr=None)
    assert sr == 24000
    duration = y.shape[0] / sr
    alignments = item["alignments"]
    try:
        alignments, gaps = match_alignment(
            tokens, alignments, duration=duration, sample_rate=sr, hop_length=256, return_full_ali=True
        )
    except Exception as e:
        return None, None, e.args[0]

    alignments = [AlignmentItem(symbol=t, start=s, duration=e - s) for t, s, e in alignments]

    cut = MonoCut(
        id=item["id"],
        start=0,
        duration=duration,
        channel=0,
        supervisions=[
            SupervisionSegment(
                id=item["id"],
                recording_id=item["id"],
                start=0,
                duration=duration,
                channel=0,
                text=None,
                alignment={"char": alignments},
            )
        ],
        recording=Recording(
            id=item["id"],
            sources=[AudioSource(source=audio_path, type="file", channels=[0])],
            sampling_rate=sr,
            num_samples=y.shape[0],
            duration=duration,
            channel_ids=[0],
        ),
        custom=dict(tokens=tokens),
    )
    return cut.to_dict(), gaps, None


def prepare_for_lhotse(ali_path: str, output_path: str):
    all_gaps = []
    errors = []

    with jsonlines.open(output_path, "w") as f_out:
        with open(ali_path) as f_in:
            N = sum([1 for _ in tqdm(f_in, desc="Scanning")])

        def worker_input():
            for item in f_in:
                yield dict(item=item)

        with jsonlines.open(ali_path) as f_in:
            for cut, gaps, err in execute_parallel(worker, worker_input(), inputs_count=N, max_workers=16, batch_size=20000):
                if err is None:
                    all_gaps.extend(gaps)
                    f_out.write(cut)
                else:
                    errors.append(err)

    print("gaps =", len(all_gaps))
    for g in all_gaps[:10]:
        print(g)

    print("error =", Counter(errors))


def test_lhotse():
    from torch.utils.data import DataLoader
    from lhotse.dataset import DynamicBucketingSampler, SpeechSynthesisDataset, OnTheFlyFeatures
    from f5_tts.model.features import VocosMelSpectrogram

    if False:
        cuts: CutSet = CutSet.from_jsonl_lazy("data/vi/test_with_alignments_lhotse.jsonl")
        cuts.compute_and_store_features(VocosMelSpectrogram(), "data/vi/test_with_alignments_lhotse").to_file(
            "data/vi/test_with_alignments_lhotse_computed.jsonl",
        )
        return

    cuts: CutSet = CutSet.from_jsonl_lazy("data/vi/test_with_alignments_lhotse_computed.jsonl")
    cuts[0].plot_alignment("char")

    sampler = DynamicBucketingSampler(cuts, max_duration=100, num_buckets=60, shuffle=True)
    dataset = SpeechSynthesisDataset(return_text=False, return_tokens=True)
    print(dataset[0])

    dl = DataLoader(dataset, sampler=sampler, batch_size=None)
    for d in dl:
        print(d)
        break


def prepare_for_vad(ali_path: str, output_path: str):
    with open(ali_path) as f_in:
        N = sum([1 for _ in tqdm(f_in, desc="Scanning")])

    skip_gap = 0
    error = []
    with jsonlines.open(ali_path) as f_in, jsonlines.open(output_path, "w") as f_out:
        for item in tqdm(f_in, total=N):
            ali = item["alignments"]
            text = item["text"]
            text = text.replace("<sp>", " ")
            text = " ".join(text.split())
            tokens = list(text)
            try:
                ali, gaps = match_alignment(tokens, ali, None, None, None, expand_gap=True, return_full_ali=True)
            except Exception as e:
                error.append(e.args[0])
                continue

            if len(gaps) > 0:
                skip_gap += 1
                continue

            audio_path = item["audio_filepath"]
            assert os.path.exists(audio_path)

            speech_ts = []
            is_sil = True
            for t, s, e in ali:
                if t in string.punctuation or t in [" ", "sp"]:
                    is_sil = True
                else:
                    if is_sil is True:
                        speech_ts.append(dict(start=s, end=e))
                    else:
                        speech_ts[-1]["end"] = e
                    is_sil = False

            f_out.write(dict(audio_path=audio_path, speech_ts=speech_ts))
            if audio_path == "/data/longnh/data/Vivoice/audio_refined/c10a9998244f55056400d65ad0cebb1160b042274ee7f038e25dd9b4c44e3ebd.wav":
                y, sr = librosa.load(audio_path, sr=None)
                plot_image(item["alignments"], y, sr)

    print("skip_gap =", skip_gap)
    print("error =", Counter(error))


def eda(data_path: str):
    gaps = []
    durations = defaultdict(list)
    with jsonlines.open(data_path) as f:
        for item in tqdm(f):
            mel_alignments = item["mel_alignments"]
            tokens = item["text"]
            assert len(tokens) == len(mel_alignments)

            for index in range(len(mel_alignments) - 1):
                cur = mel_alignments[index]
                nxt = mel_alignments[index + 1]
                if nxt[0] - cur[1] > 0:
                    gaps.append(nxt[0] - cur[1])

            for t, (s, e) in zip(tokens, mel_alignments):
                durations[t].append((e - s) * 256 / 24000)

    print(len(gaps))
    if len(gaps) > 0:
        print(min(gaps), max(gaps))

    report = {}
    for k in sorted(durations):
        vs = np.array(durations[k])
        report[k] = {
            "min": vs.min().item(),
            "mean": vs.mean().item(),
            "q(0.98)": np.quantile(vs, 0.98).item(),
            "q(0.99)": np.quantile(vs, 0.99).item(),
            "q(0.999)": np.quantile(vs, 0.999).item(),
            "max": vs.max().item(),
            "std": vs.std().item(),
            "threshold": vs.max().item(),
        }

    import yaml

    with open(os.path.join(os.path.dirname(data_path), "phoneme_duration_statistic.yaml"), "w") as f:
        yaml.safe_dump(report, f, sort_keys=False, encoding="utf8")


def plot_image(segments, waveform, sample_rate, figure_path="temp/alignment.png"):
    fig, ax = plt.subplots(1, 1, figsize=(40, 4))

    waveform = waveform / waveform.max()
    ax.plot(waveform)

    for _index, _segment in enumerate(segments):
        x0 = int(_segment[1] * sample_rate)
        x1 = int(_segment[2] * sample_rate)
        ax.axvspan(x0, x1, alpha=0.1, color="red")
        ax.text(s=_segment[0], x=x0, y=((-1) ** _index) * 0.9, rotation=0)
    plt.savefig(figure_path)

    return fig


def plot(path: str):
    with open(path) as f:
        data = json.load(f)

    y, sr = librosa.load(data["audio_filepath"], sr=None)
    plot_image(data["alignments"], y, sr)


def plot_vad(path: str):
    with open(path) as f:
        data = json.load(f)
    y, sr = librosa.load(data["audio_path"], sr=None)
    ali = []
    for ts in data["speech_ts"]:
        ali.append(("SPEECH", ts["start"], ts["end"]))

    plot_image(ali, y, sr)


if __name__ == "__main__":
    if False:
        prepare(
            ali_path="/data/longnh/data/codes/asr_validating/pipeline/kaldi/data/features/mfcc_pitch/test/ali.jsonl",
            data_path="data/vi/test.jsonl",
            output_path="data/vi/test_with_alignments.jsonl",
        )
    if False:
        prepare(
            ali_path="/data/longnh/data/codes/asr_validating/pipeline/kaldi/data/features/mfcc_pitch/all/ali.jsonl",
            data_path="data/vi/train.jsonl",
            output_path="data/vi/train_with_alignments.jsonl",
        )
    if False:
        prepare_for_lhotse(
            ali_path="/data/longnh/data/codes/asr_validating/pipeline/kaldi/data/features/mfcc_pitch/test/ali.jsonl",
            output_path="data/vi/test_with_alignments_lhotse.jsonl",
        )
    if False:
        prepare_for_vad(
            ali_path="/data/longnh/data/codes/asr_validating/pipeline/kaldi/data/features/mfcc_pitch/test/ali.jsonl",
            output_path="data/vi/test_vad.jsonl",
        )

    # plot_vad("temp/ts.json")
    test_lhotse()

    # eda("data/vi/test_with_alignments.jsonl")
    # eda("data/vi/train_with_alignments.jsonl")
