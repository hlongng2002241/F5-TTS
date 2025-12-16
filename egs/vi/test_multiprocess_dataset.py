#!/usr/bin/env python3
"""
Test script to verify multiprocess-safe IndexedDataset with PyTorch DataLoader
"""

import sys
from torch.utils.data import DataLoader, Dataset

from quick_utils.common.indexed_dataset import IndexedDataset


class IndexedDatasetWrapper(Dataset):
    """Wrapper to use IndexedDataset with PyTorch DataLoader"""

    def __init__(self, path: str):
        self.dataset = IndexedDataset(path, num_cache=32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def custom_collate(batch):
    """Custom collate function that handles variable-length data"""
    # Since each item is a dictionary, we'll just return the list of dicts
    return batch


def test_multiprocess_loading():
    """Test loading data with multiple worker processes"""

    dataset_path = "data/vi/test_with_alignments"

    print(f"Testing multiprocess loading with IndexedDataset: {dataset_path}")
    print("=" * 60)

    # Create dataset wrapper
    dataset = IndexedDatasetWrapper(dataset_path)
    print(f"Dataset size: {len(dataset)} items")

    # Test with single worker first (baseline)
    print("\n1. Testing with single worker (num_workers=0)...")
    loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=False, collate_fn=custom_collate)

    try:
        for i, batch in enumerate(loader):
            if i == 0:
                print(f"   ✓ First batch loaded successfully")
                print(f"   Batch keys: {list(batch[0].keys())}")
                print(f"   Batch size: {len(batch)}")
            if i >= 5:  # Test first 5 batches
                break
        print("   ✓ Single worker test passed!")
    except Exception as e:
        print(f"   ✗ Single worker test failed: {e}")
        return False

    # Test with multiple workers
    print("\n2. Testing with multiple workers (num_workers=4)...")
    loader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=True, collate_fn=custom_collate)

    try:
        for i, batch in enumerate(loader):
            if i == 0:
                print(f"   ✓ First batch loaded successfully")
                print(f"   Batch size: {len(batch)}")
            if i >= 10:  # Test first 10 batches
                break
        print("   ✓ Multi-worker test passed!")
    except Exception as e:
        print(f"   ✗ Multi-worker test failed: {e}")
        return False

    # Stress test with more workers
    print("\n3. Stress testing with 8 workers...")
    loader = DataLoader(dataset, batch_size=16, num_workers=8, shuffle=True, collate_fn=custom_collate)

    try:
        total_items = 0
        for i, batch in enumerate(loader):
            total_items += len(batch)
            if i >= 20:  # Test first 20 batches
                break
        print(f"   ✓ Stress test passed! Loaded {total_items} items across {i+1} batches")
    except Exception as e:
        print(f"   ✗ Stress test failed: {e}")
        return False

    # Test random access from multiple workers
    print("\n4. Testing random access patterns...")
    loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True, drop_last=True, collate_fn=custom_collate)

    try:
        batch_count = 0
        total_items = 0
        for batch in loader:
            batch_count += 1
            total_items += len(batch)
            if batch_count >= 50:  # Test 50 random batches
                break
        print(f"   ✓ Random access test passed! {total_items} items in {batch_count} batches")
    except Exception as e:
        print(f"   ✗ Random access test failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ All multiprocess tests passed successfully!")
    print("The IndexedDataset is now multiprocess-safe and works with PyTorch DataLoader!")
    return True


if __name__ == "__main__":
    success = test_multiprocess_loading()
    sys.exit(0 if success else 1)