# ruff: noqa: F722 F821

from __future__ import annotations

import os
import random
from typing import Any
from collections import defaultdict
from importlib.resources import files

import jieba
import torch
from pypinyin import Style, lazy_pinyin
from torch.nn.utils.rnn import pad_sequence


# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d) -> Any:
    return v if exists(v) else d


def is_package_available(package_name: str) -> bool:
    try:
        import importlib

        package_exists = importlib.util.find_spec(package_name) is not None
        return package_exists
    except Exception:
        return False


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


# Get tokenizer


def get_tokenizer(dataset_name, tokenizer: str = "pinyin"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char"]:
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    return vocab_char_map, vocab_size


# convert char to pinyin


def convert_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False


# get the empirically pruned step for sampling


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)


# MAS (Monotonic Alignment Search) adaptation utilities


def update_mas_alpha(model, global_step, warmup_steps=150000):
    """
    Gradually increase MAS alpha from 0 to 1 over warmup_steps.

    Args:
        model: CFM model with transformer.text_embed.mas_alpha buffer
        global_step: current training step
        warmup_steps: number of steps to reach full MAS (alpha=1)
    """
    alpha = min(1.0, global_step / warmup_steps)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'text_embed'):
        if hasattr(model.transformer.text_embed, 'mas_alpha'):
            model.transformer.text_embed.mas_alpha.fill_(alpha)
    return alpha


def update_mas_temperature(model, global_step, warmup_steps=100000):
    """
    Gradually decrease MAS temperature from 10 to 1 over warmup_steps.

    Args:
        model: CFM model with transformer.text_embed.mas_temperature buffer
        global_step: current training step
        warmup_steps: number of steps to reach sharp attention (temp=1)
    """
    progress = min(1.0, global_step / warmup_steps)
    temperature = 10.0 * (1 - progress) + 1.0 * progress
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'text_embed'):
        if hasattr(model.transformer.text_embed, 'mas_temperature'):
            model.transformer.text_embed.mas_temperature.fill_(temperature)
    return temperature


def load_checkpoint_with_mas(checkpoint_path, model, enable_mas=True):
    """
    Load V0/V1 checkpoint into model and optionally enable MAS components.

    This function:
    1. Loads V0/V1 checkpoint weights
    2. If enable_mas=True, initializes MAS components (similarity_proj, mas_alpha, mas_temperature, mel_feature_proj)

    Args:
        checkpoint_path: path to V0/V1 checkpoint
        model: CFM model (with or without MAS support)
        enable_mas: whether to initialize MAS components after loading

    Returns:
        model: model with loaded weights and optionally initialized MAS components
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'ema_model_state_dict' in checkpoint:
        # Extract and clean EMA state dict
        state_dict = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint['ema_model_state_dict'].items()
            if k not in ["initted", "step"]
        }
    else:
        state_dict = checkpoint

    # Load V0/V1 weights (MAS components don't exist in checkpoint, so strict=True works)
    model.load_state_dict(state_dict, strict=True)
    print(f"✓ Loaded V0/V1 checkpoint from: {checkpoint_path}")

    # Initialize MAS components if requested
    if enable_mas:
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'text_embed'):
            text_dim = model.transformer.text_embed.text_embed.embedding_dim
            model.transformer.text_embed.initialize_mas_components(text_dim)
            print(f"✓ Initialized MAS components (text_dim={text_dim})")

        # Initialize mel_feature_proj if model has use_mas=True
        if hasattr(model, 'use_mas') and model.use_mas and model.mel_feature_proj is None:
            num_channels = model.num_channels
            text_dim = model.transformer.text_embed.text_embed.embedding_dim
            model.mel_feature_proj = torch.nn.Linear(num_channels, text_dim)
            torch.nn.init.xavier_uniform_(model.mel_feature_proj.weight)
            torch.nn.init.zeros_(model.mel_feature_proj.bias)
            print(f"✓ Initialized mel_feature_proj ({num_channels} → {text_dim})")

    return model
