import os
import jsonlines
from tqdm import tqdm
from collections import Counter, defaultdict

import torch
import numpy as np


def match_alignment(tokens: list[str], alignments: list, duration: float, sample_rate: int, hop_length: int, expand_gap=True):
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
        assert e <= mel_len, 30
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


def prepare_attn(mel_alignments: list, mel_length: int):
    attn = torch.zeros((len(mel_alignments), mel_length)).float()
    for i_text, (s, e) in enumerate(mel_alignments):
        attn[i_text, s:e] = 1.0
    return attn


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
                # print(prepare_attn(item["mel_alignments"], int(item["duration"] * sample_rate / hop_length)))
                # return
                f_out.write(item)
            except Exception as e:
                error.append(e.args[0])

    print("gaps =", len(gaps))
    for g in gaps[:10]:
        print(g)

    print("missing =", missing)
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

    # eda("data/vi/test_with_alignments.jsonl")
    # eda("data/vi/train_with_alignments.jsonl")
