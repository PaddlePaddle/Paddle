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

import gc
import struct
import unittest

import numpy as np

import paddle


class TestCUDAVMMV2Allocator(unittest.TestCase):
    def test_allocator_facade_creates_vmm_v2_allocator(self):
        flags = paddle.get_flags(
            [
                "FLAGS_allocator_strategy",
                "FLAGS_use_vmm_auto_growth_best_fit_allocator_v2",
                "FLAGS_vmm_v2_large_pool_handle_size_in_mb",
            ]
        )
        self.assertEqual(flags["FLAGS_allocator_strategy"], "auto_growth")
        self.assertTrue(
            flags["FLAGS_use_vmm_auto_growth_best_fit_allocator_v2"]
        )
        self.assertGreater(
            flags["FLAGS_vmm_v2_large_pool_handle_size_in_mb"], 0
        )

        x = paddle.zeros([1024], dtype="float32")
        paddle.device.synchronize()
        self.assertEqual(x.shape, [1024])
        self.assertGreater(paddle.device.cuda.memory_reserved(), 0)

    def test_public_api_allocate_free_path(self):
        # Exercise the real allocator stack instead of constructing allocator
        # internals directly:
        #   AllocatorFacade -> RetryAllocator -> StreamSafeCUDAAllocator
        #   -> VMMAutoGrowthBestFitMultiPoolAllocatorV2.
        # Explicitly dropping tensors covers the stream-safe free path that
        # records VMM v2 remap-safety metadata for later OOM compaction.
        tensors = [
            paddle.zeros([1024 * 1024], dtype="float32") for _ in range(4)
        ]
        paddle.device.synchronize()
        reserved_before_free = paddle.device.cuda.memory_reserved()
        self.assertGreater(reserved_before_free, 0)

        del tensors
        gc.collect()
        paddle.device.synchronize()

        y = paddle.zeros([1024 * 1024], dtype="float32")
        paddle.device.synchronize()
        self.assertEqual(y.shape, [1024 * 1024])

    def test_ipc_round_trip_reuses_export_fd(self):
        tensor = paddle.arange(64, dtype="float32").reshape([8, 8])
        dense = tensor.value().get_tensor()
        first_meta = dense._share_cuda()
        second_meta = dense._share_cuda()

        # VMM v2 caches one exported FD per backing handle. Re-exporting the
        # same live tensor must produce the same descriptor payload.
        self.assertEqual(first_meta[0], second_meta[0])
        rebuilt = paddle.base.core.DenseTensor._new_shared_cuda(first_meta)
        np.testing.assert_array_equal(
            paddle.to_tensor(rebuilt).numpy(), tensor.numpy()
        )

    def test_ipc_rejects_malformed_payload(self):
        tensor = paddle.arange(16, dtype="float32")
        meta = list(tensor.value().get_tensor()._share_cuda())

        header = bytearray(meta[0])
        header[0] = 0xFF
        invalid_version = (bytes(header), *meta[1:])
        with self.assertRaises(ValueError):
            paddle.base.core.DenseTensor._new_shared_cuda(invalid_version)

        truncated = (meta[0][:-1], *meta[1:])
        with self.assertRaises(ValueError):
            paddle.base.core.DenseTensor._new_shared_cuda(truncated)

        invalid_dims = (meta[0], meta[1], [1024], *meta[3:])
        with self.assertRaises(ValueError):
            paddle.base.core.DenseTensor._new_shared_cuda(invalid_dims)

        version, flags, pid, entries, _, _, _ = struct.unpack_from(
            "<BHIIQQQ", meta[0], 0
        )
        self.assertEqual(version, 1)
        self.assertEqual(flags, 1)
        self.assertGreater(pid, 0)
        self.assertGreater(entries, 0)


if __name__ == "__main__":
    unittest.main()
