import os
import jsonlines
from tqdm import tqdm
from collections import Counter, defaultdict

import numpy as np
from f5_tts.model.utils import match_alignment


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
