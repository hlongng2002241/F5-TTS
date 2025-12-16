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
    """
    True is masked, otherwise False
    """
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def mask_from_alignments(mel_attn: torch.LongTensor, frac_lengths: torch.FloatTensor):
    """
    Create a mask based on alignment, masking complete characters.

    Args:
        mel_attn: Alignment matrix [batch, num_chars, mel_frames]
        frac_lengths: Fraction of characters to mask [batch]

    Returns:
        mask: Boolean mask [batch, mel_frames]
    """
    batch_size, num_chars, mel_frames = mel_attn.shape
    device = mel_attn.device

    # Create mask for each sample in batch
    mask = torch.zeros((batch_size, mel_frames), dtype=torch.bool, device=device)

    for b in range(batch_size):
        # Find characters that have aligned frames (non-zero duration)
        char_has_frames = mel_attn[b].sum(dim=1) > 0  # [num_chars]
        valid_char_indices = torch.where(char_has_frames)[0]

        if len(valid_char_indices) == 0:
            # Fallback: if no valid characters, skip this sample
            continue

        # Calculate how many characters to mask
        num_valid_chars = len(valid_char_indices)
        num_to_mask = int(frac_lengths[b] * num_valid_chars)
        num_to_mask = max(1, num_to_mask)  # At least mask 1 character

        # Randomly select a contiguous span of characters to mask
        if num_to_mask < num_valid_chars:
            # Select random starting position for the span
            max_start = num_valid_chars - num_to_mask
            start_idx = torch.randint(0, max_start + 1, (1,), device=device).item()
            selected_indices = valid_char_indices[start_idx:start_idx + num_to_mask]
        else:
            # Mask all valid characters
            selected_indices = valid_char_indices

        # Find the frame range that covers all selected characters (including gaps)
        # Get frames for each selected character
        frames_mask = mel_attn[b, selected_indices, :].sum(dim=0) > 0

        # Find the start and end of the mask region to make it contiguous
        frame_indices = torch.where(frames_mask)[0]
        if len(frame_indices) > 0:
            start_frame = frame_indices[0].item()
            end_frame = frame_indices[-1].item() + 1  # +1 for exclusive end

            # Create a contiguous mask from start to end (includes any gaps)
            mask[b, start_frame:end_frame] = True

    return mask


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


# alignment


def match_alignment(tokens: list[str], alignments: list, duration: float, sample_rate: int, hop_length: int, expand_gap=True, return_full_ali=False):
    aligned_tokens = [a[0] for a in alignments if a[0] != "sp"]
    for a in aligned_tokens:
        assert a.strip() != "", 1

    text_tokens_non_space = [t for t in tokens if t != " "]
    # print(aligned_tokens)
    # print(text_tokens_non_space)
    assert aligned_tokens == text_tokens_non_space, 2
    assert len(aligned_tokens) <= len(tokens), 3

    i_align = 0
    full_alignments = []
    for i_text in range(len(tokens)):
        t_token = tokens[i_text]
        if t_token == " ":
            if alignments[i_align][0] == "sp":
                full_alignments.append((" ", alignments[i_align][1], alignments[i_align][2]))
                i_align += 1
            else:
                assert tokens[i_text - 1] == alignments[i_align - 1][0], 4
                assert tokens[i_text + 1] == alignments[i_align][0], 5
                full_alignments.append((" ", alignments[i_align - 1][2], alignments[i_align][1]))
        else:
            assert t_token == alignments[i_align][0], 6
            full_alignments.append(alignments[i_align])
            i_align += 1

    for index in range(1, len(full_alignments)):
        prev = full_alignments[index - 1]
        cur = full_alignments[index]
        assert prev[2] <= cur[1], 7

    if expand_gap:
        full_alignments, unexpanded_gaps = expand_alignment_gaps(full_alignments)
    else:
        unexpanded_gaps = []
        for index in range(len(full_alignments) - 1):
            cur = full_alignments[index]
            nxt = full_alignments[index + 1]
            if nxt[1] - cur[2] > 0:
                unexpanded_gaps.append((nxt[1] - cur[2], cur, nxt))

    for index in range(1, len(full_alignments)):
        prev = full_alignments[index - 1]
        cur = full_alignments[index]
        assert prev[2] <= cur[1], 8
        
    if return_full_ali:
        return full_alignments, unexpanded_gaps

    mel_len = int(duration * sample_rate / hop_length)
    mel_alignments = []
    mel_durations = []
    for t, s, e in full_alignments:
        # based on statistic, remove sample that have " " lasting longer than 2 seconds
        if t == " " and e - s > 2:
            raise ValueError(10)
            
        s = round(s * sample_rate / hop_length)
        e = round(e * sample_rate / hop_length)
        assert s <= e, 20
        assert e <= mel_len, (30, e, mel_len)
        mel_durations.append(e - s)
        if e - s == 0:
            assert t == " ", 40
        mel_alignments.append((s, e))

    assert len(mel_durations) == len(tokens), 50
    assert sum(mel_durations) <= mel_len, 60
    assert len(mel_alignments) == len(tokens), 70

    return mel_alignments, unexpanded_gaps


def expand_alignment_gaps(full_alignments: list[tuple[str, float, float]]):
    full_alignments = [list(ali) for ali in full_alignments]
    removed_indexes = []
    unexpanded_gaps = []

    for index in range(len(full_alignments) - 1):
        cur = full_alignments[index]
        nxt = full_alignments[index + 1]
        if nxt[1] - cur[2] > 0:
            if cur[0] == " " and nxt[0] == " ":
                nxt[1] = cur[1]
                removed_indexes.append(index)

            elif cur[0] == " ":
                cur[2] = nxt[1]

            elif nxt[0] == " ":
                nxt[1] = cur[2]

            else:
                unexpanded_gaps.append((nxt[1] - cur[2], cur, nxt))

    full_alignments = [tuple(ali) for index, ali in enumerate(full_alignments) if index not in removed_indexes]

    return full_alignments, unexpanded_gaps
