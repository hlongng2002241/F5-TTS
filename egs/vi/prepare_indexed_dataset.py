#!/usr/bin/env python3
"""
Convert JSONL files with alignments to indexed datasets for efficient loading.
"""

import json
import os
import sys
from tqdm import tqdm

from quick_utils.common.indexed_dataset import IndexedDatasetBuilder, IndexedDataset


def convert_jsonl_to_indexed_dataset(input_path: str, output_path: str) -> None:
    """
    Convert a JSONL file to an indexed dataset.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output indexed dataset (without extension)
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Count total lines for progress bar
    print(f"Counting lines in {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    print(f"Converting {total_lines} entries from {input_path} to {output_path}...")

    # Convert JSONL to indexed dataset
    with IndexedDatasetBuilder(output_path) as builder:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=total_lines, desc="Processing"):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Parse JSON line
                    item = json.loads(line)
                    # Add to indexed dataset
                    builder.add_item(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line: {e}")
                    continue

    # Verify the dataset was created successfully
    if IndexedDataset.exist_dataset(os.path.dirname(output_path) or ".", os.path.basename(output_path)):
        print(f"✓ Successfully created indexed dataset: {output_path}")

        # Print dataset statistics
        dataset = IndexedDataset(output_path, num_cache=0)
        print(f"  Total entries: {len(dataset)}")

        # Show sample entry
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Sample entry keys: {list(sample.keys()) if isinstance(sample, dict) else 'N/A'}")
            dur = 0
            for index in tqdm(range(len(dataset)), total=len(dataset)):
                dur += dataset[index]["duration"]
            print(dur)

        dataset.close()
    else:
        print(f"✗ Failed to create indexed dataset: {output_path}")
        sys.exit(1)


def main():
    convert_jsonl_to_indexed_dataset("data/vi/train_with_alignments.jsonl", "data/vi/train_with_alignments")
    convert_jsonl_to_indexed_dataset("data/vi/test_with_alignments.jsonl", "data/vi/test_with_alignments")


if __name__ == "__main__":
    main()
