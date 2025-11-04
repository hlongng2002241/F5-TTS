import jsonlines
from tqdm import tqdm
from collections import Counter
import torch


def match_alignment(tokens: list[str], alignments: list, duration: float, sample_rate: int, hop_length: int):
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
                full_alignments.append(alignments[i_align])
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

    mel_len = int(duration * sample_rate / hop_length)
    mel_alignments = []
    durations = []
    for _, s, e in full_alignments:
        s = round(s * sample_rate / hop_length)
        e = round(e * sample_rate / hop_length)
        assert s <= e, 8
        assert e <= mel_len, 9
        durations.append(e - s)
        mel_alignments.append((s, e))

    assert len(durations) == len(tokens), 10
    assert sum(durations) <= mel_len, 11
    assert len(mel_alignments) == len(tokens), 12

    return mel_alignments


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
    with jsonlines.open(output_path, "w") as f_out, jsonlines.open(data_path) as f_in:
        for item in tqdm(f_in):
            if item["id"] not in item_id_to_ali:
                missing += 1
                continue
            ali = item_id_to_ali[item["id"]]
            assert item["audio_path"] == ali["audio_filepath"]
            text = [t.lower() for t in item["text"]]
            try:
                item["mel_alignments"] = match_alignment(
                    text, ali["alignments"], item["duration"], sample_rate=sample_rate, hop_length=hop_length
                )
                # print(prepare_attn(item["mel_alignments"], int(item["duration"] * sample_rate / hop_length)))
                # return
                f_out.write(item)
            except Exception as e:
                error.append(e.args[0])

    print("missing =", missing)
    print("error =", Counter(error))


if __name__ == "__main__":
    if True:
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
