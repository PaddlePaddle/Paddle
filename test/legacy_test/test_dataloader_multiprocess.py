#   Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import os
import subprocess
import sys
import unittest
from collections import Counter

import numpy as np

import paddle
from paddle.io import DataLoader, Dataset, IterableDataset


class RandomDataset(Dataset):
    def __init__(self, sample_num, feature_dim=10):
        self.sample_num = sample_num
        self.feature_dim = feature_dim
        # The first feature holds the sample index, so that every sample can
        # be identified after having been sent through the worker queues.
        self.data = np.random.rand(sample_num, feature_dim).astype('float32')
        self.data[:, 0] = np.arange(sample_num)
        self.label = np.random.randint(0, 2, size=(sample_num, 1)).astype(
            'int64'
        )

    def __getitem__(self, idx):
        return (
            paddle.to_tensor(self.data[idx]),
            paddle.to_tensor(self.label[idx]),
        )

    def __len__(self):
        return self.sample_num


class RangeIterableDataset(IterableDataset):
    def __init__(self, sample_num, feature_dim=10):
        self.sample_num = sample_num
        self.feature_dim = feature_dim

    def __iter__(self):
        for i in range(self.sample_num):
            yield np.full([self.feature_dim], i, dtype='float32')


class TestDataLoaderMultiprocess(unittest.TestCase):
    def setUp(self):
        self.sample_num = 100
        self.batch_size = 8
        self.feature_dim = 10
        self.epoch_num = 3

    def run_simple_net(
        self, num_workers, use_shared_memory=False, persistent_workers=False
    ):
        paddle.seed(2026)
        np.random.seed(2026)

        dataset = RandomDataset(self.sample_num, self.feature_dim)
        loader = DataLoader(
            dataset,
            places=paddle.CPUPlace(),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            use_shared_memory=use_shared_memory,
            persistent_workers=persistent_workers,
        )

        # NOTE: the loss is summed (not averaged) over the samples so that the
        # result does not depend on how the samples are grouped into batches
        # nor on the order in which they are visited. The shuffle order is not
        # reproducible across num_workers settings anyway, because the
        # multi-process iterator draws the worker base seed from the same numpy
        # random state as the sampler.
        loss_sum = 0.0
        for epoch in range(self.epoch_num):
            seen = []
            for data, label in loader():
                self.assertEqual(list(data.shape[1:]), [self.feature_dim])
                seen.extend(np.array(data)[:, 0].astype('int64').tolist())
                # Simple linear layer
                pred = paddle.mean(data, axis=1, keepdim=True)
                loss = paddle.nn.functional.binary_cross_entropy_with_logits(
                    pred, paddle.cast(label, 'float32'), reduction='sum'
                )
                loss_sum += float(loss)
            # every epoch must yield the whole dataset, each sample once
            self.assertEqual(sorted(seen), list(range(self.sample_num)))
        return loss_sum / (self.epoch_num * self.sample_num)

    def check_against_single_process(self, **kwargs):
        loss_single = self.run_simple_net(num_workers=0)
        loss_multi = self.run_simple_net(**kwargs)
        # The samples are the same and only their order differs, so the mean
        # loss over whole epochs must match up to accumulation order.
        np.testing.assert_allclose(loss_multi, loss_single, rtol=1e-5)

    def test_multiprocess_singleprocess_loss_close(self):
        """
        Test that multi-process (num_workers=2) and single-process
        (num_workers=0) load the same data and produce the same loss.
        """
        self.check_against_single_process(num_workers=2)

    def test_multiprocess_with_shared_memory(self):
        """
        Test multi-process DataLoader with use_shared_memory=True.
        """
        self.check_against_single_process(num_workers=2, use_shared_memory=True)

    def test_multiprocess_more_workers(self):
        """
        Test with num_workers=4 to stress-test the worker pool.
        """
        self.check_against_single_process(num_workers=4)

    def test_multiprocess_persistent_workers(self):
        """
        Test with persistent_workers=True to ensure worker reuse works.
        """
        self.check_against_single_process(
            num_workers=2, persistent_workers=True
        )

    def test_multiprocess_iterable_dataset(self):
        """
        Test multi-process DataLoader with IterableDataset. Without a
        worker_init_fn splitting the data, every worker iterates the whole
        dataset, so each sample is expected to appear num_workers times.
        """
        sample_num = 50
        num_workers = 2
        dataset = RangeIterableDataset(sample_num, self.feature_dim)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
        )
        seen = []
        # every sample is a single array, so the loader yields the collated
        # tensor directly instead of a list of fields
        for batch in loader():
            self.assertEqual(list(batch.shape[1:]), [self.feature_dim])
            seen.extend(np.array(batch)[:, 0].astype('int64').tolist())

        counter = Counter(seen)
        self.assertEqual(sorted(counter.keys()), list(range(sample_num)))
        self.assertEqual(set(counter.values()), {num_workers})


@unittest.skipIf(
    sys.platform not in ('linux', 'win32'),
    "shared-memory tensor reduction only supported on Linux/Windows",
)
class TestDenseTensorReduction(unittest.TestCase):
    """Covers the ForkingPickler reducers used by the DataLoader worker
    result queue: _share_filename / _new_shared_filename / incref / decref."""

    def test_reducer_registered(self):
        from multiprocessing.reduction import ForkingPickler

        from paddle.incubate.multiprocessing import init_reductions

        init_reductions()
        self.assertIn(
            paddle.base.core.DenseTensor, ForkingPickler._extra_reducers
        )

    def test_share_filename_roundtrip(self):
        import pickle
        from multiprocessing.reduction import ForkingPickler

        from paddle.incubate.multiprocessing import init_reductions

        init_reductions()
        arr = np.arange(64, dtype='float32').reshape(8, 8)
        tensor = paddle.base.core.DenseTensor()
        tensor.set(arr, paddle.base.core.CPUPlace())

        data = bytes(ForkingPickler.dumps(tensor))
        rebuilt = pickle.loads(data)
        np.testing.assert_array_equal(np.array(rebuilt), arr)

    def test_share_filename_roundtrip_multiple(self):
        # Repeated roundtrips exercise handle/section reclamation
        # (WindowsHandleKeeper sweep on Windows, shm_unlink on Linux).
        import pickle
        from multiprocessing.reduction import ForkingPickler

        from paddle.incubate.multiprocessing import init_reductions

        init_reductions()
        for i in range(20):
            arr = np.full([16], i, dtype='int64')
            tensor = paddle.base.core.DenseTensor()
            tensor.set(arr, paddle.base.core.CPUPlace())
            rebuilt = pickle.loads(bytes(ForkingPickler.dumps(tensor)))
            np.testing.assert_array_equal(np.array(rebuilt), arr)

    def test_sweep_mmap_handles_smoke(self):
        # No-op on Linux; on Windows reclaims refcount==0 keeper entries.
        # Must be callable at any time without pending mappings.
        paddle.base.core._sweep_mmap_handles()


class TestParentWatchDog(unittest.TestCase):
    def test_alive_parent(self):
        from paddle.io.dataloader.worker import ParentWatchDog

        # On POSIX is_alive() compares os.getppid() against the stored pid,
        # so use the real parent; on Windows it queries the process state,
        # so our own (running) pid works.
        parent = os.getpid() if sys.platform == 'win32' else os.getppid()
        wd = ParentWatchDog(parent)
        self.assertTrue(wd.is_alive())

    @unittest.skipIf(
        sys.platform != 'win32', "Windows-specific parent liveness check"
    )
    def test_dead_parent(self):
        from paddle.io.dataloader.worker import ParentWatchDog

        p = subprocess.Popen([sys.executable, '-c', 'pass'])
        p.wait()
        # p's handle is still open, so the pid is not reused yet and
        # OpenProcess sees an exited process.
        wd = ParentWatchDog(p.pid)
        self.assertFalse(wd.is_alive())

    @unittest.skipIf(
        sys.platform != 'win32', "Windows-specific parent liveness check"
    )
    def test_nonexistent_parent(self):
        from paddle.io.dataloader.worker import ParentWatchDog

        wd = ParentWatchDog(0x7FFFFFF)
        self.assertFalse(wd.is_alive())


if __name__ == '__main__':
    unittest.main()
