# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test that DataLoader supports returning non-Tensor data (Issue #77754)."""

import unittest

from paddle.io import DataLoader, Dataset


class NonTensorDataset(Dataset):
    """Dataset that returns only non-Tensor data (strings)."""

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return "a"

    def __len__(self):
        return self.num_samples


class MixedDataset(Dataset):
    """Dataset that returns a mix of Tensor and non-Tensor data."""

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        import numpy as np

        return "item_" + str(idx), np.array([idx], dtype="float32")

    def __len__(self):
        return self.num_samples


def identity_collate(batch):
    return batch


class TestNonTensorDataLoader(unittest.TestCase):
    """Test DataLoader with non-Tensor return values (Issue #77754)."""

    def test_single_process_non_tensor(self):
        """Single-process DataLoader should iterate all batches for non-Tensor data."""
        dataset = NonTensorDataset(20)
        loader = DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            drop_last=True,
            collate_fn=identity_collate,
            num_workers=0,
        )
        batches = []
        for batch in loader:
            batches.append(batch)

        # With drop_last=True, 20 samples / batch_size=10 = 2 full batches
        self.assertEqual(len(batches), 2)
        for batch in batches:
            self.assertEqual(len(batch), 10)
            for item in batch:
                self.assertEqual(item, "a")

    def test_single_process_non_tensor_no_drop_last(self):
        """Single-process DataLoader without drop_last for non-Tensor data."""
        dataset = NonTensorDataset(25)
        loader = DataLoader(
            dataset,
            batch_size=10,
            shuffle=False,
            drop_last=False,
            collate_fn=identity_collate,
            num_workers=0,
        )
        batches = []
        for batch in loader:
            batches.append(batch)

        # 25 samples / batch_size=10 = 2 full + 1 partial (5 items)
        self.assertEqual(len(batches), 3)
        self.assertEqual(len(batches[0]), 10)
        self.assertEqual(len(batches[1]), 10)
        self.assertEqual(len(batches[2]), 5)

    def test_multi_process_non_tensor(self):
        """Multi-process DataLoader should iterate all batches for non-Tensor data."""
        dataset = NonTensorDataset(20)
        loader = DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            drop_last=True,
            collate_fn=identity_collate,
            num_workers=2,
        )
        batches = []
        for batch in loader:
            batches.append(batch)

        self.assertEqual(len(batches), 2)
        for batch in batches:
            self.assertEqual(len(batch), 10)

    def test_mixed_tensor_and_non_tensor(self):
        """DataLoader with mixed Tensor and non-Tensor data should work correctly."""
        dataset = MixedDataset(20)
        loader = DataLoader(
            dataset,
            batch_size=10,
            shuffle=False,
            drop_last=True,
            collate_fn=identity_collate,
            num_workers=0,
        )
        batches = []
        for batch in loader:
            batches.append(batch)

        self.assertEqual(len(batches), 2)
        for batch in batches:
            self.assertEqual(len(batch), 10)

    def test_multi_place_non_tensor(self):
        """Multi-place DataLoader handles all-non-Tensor batches.

        Verifies that ReadNextList() uses per-reader status instead of
        item.empty() to filter EOF, so empty TensorArrays from non-Tensor
        batches are preserved in multi-place (kKeepOrder) mode.
        """
        import paddle

        class NonTensorDS(Dataset):
            def __init__(self, n):
                self.n = n

            def __getitem__(self, idx):
                return "a"

            def __len__(self):
                return self.n

        # Multi-place with non-Tensor data — regression for ordered reader
        dataset = NonTensorDS(20)
        loader = DataLoader(
            dataset,
            batch_size=10,
            shuffle=False,
            drop_last=True,
            collate_fn=identity_collate,
            num_workers=0,
            places=[paddle.CPUPlace(), paddle.CPUPlace()],
        )
        batches = []
        for batch in loader:
            batches.append(batch)
        # 20 samples / batch_size=10 / 2 places = 1 Python iteration
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 10)
        for item in batches[0]:
            self.assertEqual(item, "a")


if __name__ == "__main__":
    unittest.main()
