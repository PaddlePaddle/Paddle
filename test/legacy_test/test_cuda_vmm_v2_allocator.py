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
import json
import struct
import unittest

import numpy as np

import paddle
import paddle.incubate.multiprocessing as mp
from paddle.device.cuda.memory_analyzer import MemoryAnalysisTool
from paddle.incubate.multiprocessing import reductions


def send_vmm_tensor(queue, release_event, device):
    paddle.set_device(device)
    tensor = paddle.arange(64, dtype="float32").reshape([8, 8])
    queue.put(tensor)
    release_event.wait(30)


class TestCUDAVMMV2Allocator(unittest.TestCase):
    @staticmethod
    def _contains_ptr(blocks, ptr, is_free=None):
        return any(
            address <= ptr < address + size
            and (is_free is None or free == is_free)
            for size, address, free in blocks
        )

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

        self.assertEqual(len(first_meta), 7)
        self.assertEqual(first_meta[2], tensor.numel() * tensor.element_size())
        self.assertEqual(first_meta[4], list(tensor.shape))
        self.assertEqual(first_meta[6], tensor.place.gpu_device_id())
        self.assertTrue(reductions._is_vmm_ipc_metadata(first_meta))
        self.assertFalse(
            reductions._is_vmm_ipc_metadata(
                (bytes(64), 0, 0, 0, [], [], first_meta[6])
            )
        )

        # VMM v2 caches one exported FD per backing handle. Re-exporting the
        # same live tensor must produce the same descriptor payload.
        self.assertEqual(first_meta[0], second_meta[0])
        rebuilt = paddle.base.core.DenseTensor._new_shared_cuda(first_meta)
        np.testing.assert_array_equal(
            paddle.to_tensor(rebuilt).numpy(), tensor.numpy()
        )

        obsolete_meta = (
            first_meta[0],
            first_meta[3],
            first_meta[4],
            first_meta[5],
            first_meta[6],
        )
        self.assertFalse(reductions._is_vmm_ipc_metadata(obsolete_meta))
        with self.assertRaises(RuntimeError):
            paddle.base.core.DenseTensor._new_shared_cuda(obsolete_meta)

        rebuild, rebuild_args = reductions._reduce_lodtensor(dense)
        self.assertIs(rebuild, reductions._rebuild_vmm_tensor_from_fds)
        multiprocessing_rebuilt = rebuild(*rebuild_args)
        np.testing.assert_array_equal(
            paddle.to_tensor(multiprocessing_rebuilt).numpy(), tensor.numpy()
        )

        serialized_meta = list(first_meta)
        serialized_meta[0] = serialized_meta[0].decode("latin-1")
        received_meta = json.loads(json.dumps(serialized_meta))
        received_meta[0] = received_meta[0].encode("latin-1")
        received_meta[6] = tensor.place.gpu_device_id()
        json_rebuilt = paddle.base.core.DenseTensor._new_shared_cuda(
            tuple(received_meta)
        )
        np.testing.assert_array_equal(
            paddle.to_tensor(json_rebuilt).numpy(), tensor.numpy()
        )

    def test_ipc_cross_process_round_trip(self):
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        release_event = ctx.Event()
        process = ctx.Process(
            target=send_vmm_tensor,
            args=(queue, release_event, paddle.device.get_device()),
        )
        process.start()
        try:
            received = queue.get(timeout=30)
            np.testing.assert_array_equal(
                received.numpy(),
                np.arange(64, dtype="float32").reshape([8, 8]),
            )
        finally:
            release_event.set()
            process.join(30)
            if process.is_alive():
                process.terminate()
                process.join()
        self.assertEqual(process.exitcode, 0)

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

        with self.assertRaises(RuntimeError):
            paddle.base.core.DenseTensor._new_shared_cuda(tuple(meta[:6]))

        invalid_dims = (*meta[:4], [1024], *meta[5:])
        with self.assertRaises(ValueError):
            paddle.base.core.DenseTensor._new_shared_cuda(invalid_dims)

        invalid_offset = (meta[0], meta[1] + 1, *meta[2:])
        with self.assertRaises(ValueError):
            paddle.base.core.DenseTensor._new_shared_cuda(invalid_offset)

        invalid_size = (meta[0], meta[1], meta[2] + 1, *meta[3:])
        with self.assertRaises(ValueError):
            paddle.base.core.DenseTensor._new_shared_cuda(invalid_size)

        version, flags, pid, entries, _, _, _ = struct.unpack_from(
            "<BHIIQQQ", meta[0], 0
        )
        self.assertEqual(version, 1)
        self.assertEqual(flags, 1)
        self.assertGreater(pid, 0)
        self.assertGreater(entries, 0)

    def test_vmm_v2_block_info(self):
        small = paddle.zeros([1024], dtype="float32")
        large = paddle.zeros([1024 * 1024], dtype="float32")
        paddle.device.synchronize()
        small_ptr = small.data_ptr()
        large_ptr = large.data_ptr()

        small_info = MemoryAnalysisTool.vmm_small_all_block_info()
        large_info = MemoryAnalysisTool.vmm_large_all_block_info()
        all_info = MemoryAnalysisTool.vmm_all_block_info()

        self.assertEqual(len(small_info), 1)
        self.assertEqual(len(large_info), 1)
        self.assertEqual(len(all_info), 2)
        self.assertTrue(self._contains_ptr(small_info[0], small_ptr, False))
        self.assertTrue(self._contains_ptr(large_info[0], large_ptr, False))
        self.assertFalse(self._contains_ptr(small_info[0], large_ptr))
        self.assertFalse(self._contains_ptr(large_info[0], small_ptr))

        del small
        del large
        gc.collect()
        paddle.device.synchronize()

        free_info = MemoryAnalysisTool.vmm_free_block_info()
        self.assertEqual(len(free_info), 2)
        self.assertTrue(
            any(
                address <= small_ptr < address + size
                for pool in free_info
                for size, address in pool
            )
        )
        self.assertTrue(
            any(
                address <= large_ptr < address + size
                for pool in free_info
                for size, address in pool
            )
        )

    def test_vmm_v2_all_block_info_excludes_unmapped_ranges(self):
        tensors = [
            paddle.zeros([1024 * 1024], dtype="float32") for _ in range(3)
        ]
        paddle.device.synchronize()
        hole_ptr = tensors[1].data_ptr()

        del tensors[1]
        gc.collect()
        paddle.device.synchronize()
        paddle.device.cuda.empty_cache()

        all_info = MemoryAnalysisTool.vmm_all_block_info()
        self.assertFalse(
            any(self._contains_ptr(pool, hole_ptr) for pool in all_info)
        )


if __name__ == "__main__":
    unittest.main()
