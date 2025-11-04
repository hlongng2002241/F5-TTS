#!/usr/bin/env python3
"""
Test IndexedDataset with explicit multiprocessing spawn method.
This is the strictest test for multiprocessing compatibility.
"""

import sys
import os
import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as torch_mp

from quick_utils.common.indexed_dataset import IndexedDataset


def collate_fn(batch):
    """Custom collate function for variable-length data - must be at module level for pickling"""
    return batch


class IndexedDatasetWrapper(Dataset):
    """Wrapper to use IndexedDataset with PyTorch DataLoader"""

    def __init__(self, path: str):
        self.dataset = IndexedDataset(path, num_cache=32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def worker_test_func(worker_id, dataset_path, indices, return_dict):
    """Worker function for multiprocessing test"""
    try:
        # Each worker creates its own dataset instance
        dataset = IndexedDataset(dataset_path, num_cache=16)

        results = []
        for idx in indices:
            item = dataset[idx]
            results.append({
                'worker_id': worker_id,
                'index': idx,
                'id': item['id'],
                'text_len': len(item['text']),
                'pid': os.getpid()
            })

        return_dict[worker_id] = {
            'success': True,
            'results': results,
            'pid': os.getpid()
        }
    except Exception as e:
        return_dict[worker_id] = {
            'success': False,
            'error': str(e),
            'pid': os.getpid()
        }


def test_direct_multiprocessing_spawn():
    """Test with direct multiprocessing using spawn method"""

    print("=" * 60)
    print("Testing direct multiprocessing with spawn method")
    print("=" * 60)

    dataset_path = "data/vi/test_with_alignments"

    # Force spawn method (strictest test)
    mp.set_start_method('spawn', force=True)

    print(f"Multiprocessing start method: {mp.get_start_method()}")
    print(f"Dataset path: {dataset_path}")

    # Test with multiple processes
    num_workers = 4
    items_per_worker = 10

    print(f"\n1. Testing with {num_workers} processes, {items_per_worker} items each...")

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for worker_id in range(num_workers):
        # Each worker gets different indices
        indices = list(range(worker_id * items_per_worker, (worker_id + 1) * items_per_worker))

        p = mp.Process(
            target=worker_test_func,
            args=(worker_id, dataset_path, indices, return_dict)
        )
        p.start()
        processes.append(p)

    # Wait for all processes
    for p in processes:
        p.join()

    # Check results
    all_success = True
    for worker_id in range(num_workers):
        result = return_dict[worker_id]
        if result['success']:
            print(f"   Worker {worker_id} (PID {result['pid']}): ✓ Processed {len(result['results'])} items")
        else:
            print(f"   Worker {worker_id} (PID {result['pid']}): ✗ Error: {result['error']}")
            all_success = False

    if all_success:
        print("   ✓ All workers completed successfully!")
    else:
        print("   ✗ Some workers failed!")
        return False

    # Verify data integrity
    print("\n2. Verifying data integrity across processes...")
    all_items = []
    for worker_id in range(num_workers):
        all_items.extend(return_dict[worker_id]['results'])

    # Check that all indices were processed
    processed_indices = set(item['index'] for item in all_items)
    expected_indices = set(range(num_workers * items_per_worker))

    if processed_indices == expected_indices:
        print(f"   ✓ All {len(expected_indices)} indices processed correctly")
    else:
        missing = expected_indices - processed_indices
        print(f"   ✗ Missing indices: {missing}")
        return False

    # Check for unique PIDs (should have different PIDs with spawn)
    pids = set(item['pid'] for item in all_items)
    print(f"   ✓ Used {len(pids)} different process IDs: {pids}")

    return True


def test_pytorch_dataloader_spawn():
    """Test with PyTorch DataLoader using spawn multiprocessing context"""

    print("\n" + "=" * 60)
    print("Testing PyTorch DataLoader with spawn multiprocessing")
    print("=" * 60)

    # Force PyTorch to use spawn method
    torch_mp.set_start_method('spawn', force=True)

    dataset_path = "data/vi/test_with_alignments"
    dataset = IndexedDatasetWrapper(dataset_path)

    print(f"PyTorch multiprocessing start method: {torch_mp.get_start_method()}")
    print(f"Dataset size: {len(dataset)} items")

    # Test with different configurations
    test_configs = [
        {'num_workers': 2, 'batch_size': 8, 'desc': 'Small test'},
        {'num_workers': 4, 'batch_size': 16, 'desc': 'Medium test'},
        {'num_workers': 8, 'batch_size': 32, 'desc': 'Large test'},
    ]

    for i, config in enumerate(test_configs, 1):
        print(f"\n{i}. {config['desc']} (workers={config['num_workers']}, batch={config['batch_size']})...")

        loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            shuffle=True,
            collate_fn=collate_fn,
            multiprocessing_context='spawn',  # Explicitly use spawn
            persistent_workers=True  # Keep workers alive between epochs
        )

        try:
            total_batches = 0
            total_items = 0

            for batch_idx, batch in enumerate(loader):
                total_batches += 1
                total_items += len(batch)

                # Verify batch integrity
                for item in batch:
                    assert 'id' in item, "Missing 'id' field"
                    assert 'text' in item, "Missing 'text' field"
                    assert 'mel_alignments' in item, "Missing 'mel_alignments' field"

                if batch_idx >= 10:  # Test 10 batches
                    break

            print(f"   ✓ Loaded {total_items} items in {total_batches} batches")

            # Test multiple epochs (workers should persist)
            epoch2_items = 0
            for batch_idx, batch in enumerate(loader):
                epoch2_items += len(batch)
                if batch_idx >= 5:
                    break

            print(f"   ✓ Second epoch: loaded {epoch2_items} items (persistent workers)")

        except Exception as e:
            print(f"   ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n   ✓ All DataLoader spawn tests passed!")
    return True


def test_pickle_compatibility():
    """Test that IndexedDataset can be pickled and unpickled correctly"""

    print("\n" + "=" * 60)
    print("Testing pickle/unpickle compatibility")
    print("=" * 60)

    import pickle

    dataset_path = "data/vi/test_with_alignments"

    print("1. Creating original dataset...")
    original_dataset = IndexedDataset(dataset_path, num_cache=16)
    print(f"   Original dataset size: {len(original_dataset)}")

    # Get some data from original
    original_item_0 = original_dataset[0]
    original_item_100 = original_dataset[100]

    print("\n2. Pickling dataset...")
    pickled = pickle.dumps(original_dataset)
    print(f"   Pickled size: {len(pickled)} bytes")

    print("\n3. Unpickling dataset...")
    unpickled_dataset = pickle.loads(pickled)
    print(f"   Unpickled dataset size: {len(unpickled_dataset)}")

    print("\n4. Verifying unpickled dataset works correctly...")
    # The unpickled dataset should work in the new process
    unpickled_item_0 = unpickled_dataset[0]
    unpickled_item_100 = unpickled_dataset[100]

    # Verify data integrity
    assert original_item_0['id'] == unpickled_item_0['id'], "Item 0 ID mismatch"
    assert original_item_0['text'] == unpickled_item_0['text'], "Item 0 text mismatch"
    assert original_item_100['id'] == unpickled_item_100['id'], "Item 100 ID mismatch"
    assert original_item_100['text'] == unpickled_item_100['text'], "Item 100 text mismatch"

    print("   ✓ Unpickled dataset works correctly!")
    print("   ✓ Data integrity verified!")

    # Test that file handles are recreated properly
    print("\n5. Testing file handle recreation...")

    # Access multiple random indices
    import random
    random_indices = random.sample(range(len(unpickled_dataset)), 20)

    for idx in random_indices:
        item = unpickled_dataset[idx]
        assert 'id' in item and 'text' in item, f"Item {idx} missing required fields"

    print(f"   ✓ Successfully accessed {len(random_indices)} random items after unpickling")

    return True


def main():
    """Run all spawn tests"""

    print("Testing IndexedDataset with multiprocessing spawn method")
    print("This is the strictest test for multiprocessing compatibility")
    print("=" * 60)

    all_passed = True

    # Test 1: Direct multiprocessing with spawn
    try:
        if not test_direct_multiprocessing_spawn():
            all_passed = False
    except Exception as e:
        print(f"Direct multiprocessing test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 2: Pickle compatibility
    try:
        if not test_pickle_compatibility():
            all_passed = False
    except Exception as e:
        print(f"Pickle compatibility test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 3: PyTorch DataLoader with spawn
    try:
        if not test_pytorch_dataloader_spawn():
            all_passed = False
    except Exception as e:
        print(f"PyTorch DataLoader test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓✓✓ ALL SPAWN TESTS PASSED! ✓✓✓")
        print("IndexedDataset is fully compatible with multiprocessing spawn!")
    else:
        print("✗✗✗ Some tests failed ✗✗✗")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())