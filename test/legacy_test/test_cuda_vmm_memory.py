# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import platform
import unittest

import numpy as np

import paddle
from paddle.base import core
from paddle.distributed.fleet.utils.tensor_fusion_helper import (
    FusedCommBuffer,
    build_reduce_scatter_buffer,
)
from paddle.incubate.multiprocessing import reductions
from paddle.optimizer.fusion_utils import FusionStorage, FusionStorageHelper


def _skip_vmm_tests() -> bool:
    return (
        (not paddle.is_compiled_with_cuda())
        or paddle.is_compiled_with_rocm()
        or platform.system() == "Windows"
    )


_VMM_RUNTIME_AVAILABLE = None


def _vmm_runtime_available() -> bool:
    global _VMM_RUNTIME_AVAILABLE
    if _VMM_RUNTIME_AVAILABLE is not None:
        return _VMM_RUNTIME_AVAILABLE
    if _skip_vmm_tests():
        _VMM_RUNTIME_AVAILABLE = False
        return False
    try:
        tensor = paddle.randn([32], dtype="float32")
        meta = tensor.get_tensor()._share_cuda()
        rebuilt = paddle.base.core.DenseTensor._new_shared_cuda(meta)
        _ = paddle.to_tensor(rebuilt)
        _VMM_RUNTIME_AVAILABLE = True
    except Exception:
        _VMM_RUNTIME_AVAILABLE = False
    return _VMM_RUNTIME_AVAILABLE


class TestMemoryreserved(unittest.TestCase):
    def setUp(self):
        if paddle.base.is_compiled_with_cuda():
            paddle.set_flags(
                {
                    'FLAGS_use_virtual_memory_auto_growth': 1,
                }
            )

    def _simple_parameters(self):
        layer = paddle.nn.Linear(8, 4)
        return list(layer.parameters())

    def func_test_memory_stats(self):
        if core.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm():
            # 256 float32 data, with 4 bytes for each one
            alloc_size = 4 * 256
            # The chunk size of VMM allocator is aligned to granularity, which is at least 2 MB.
            reserved_size = 2 * 1024 * 1024

            tensor1 = paddle.zeros(shape=[256])
            tensor2 = paddle.zeros(shape=[256])
            self.assertEqual(
                paddle.device.cuda.memory_reserved(), reserved_size
            )
            self.assertEqual(
                paddle.device.cuda.memory_allocated(), 2 * alloc_size
            )

            del tensor1
            self.assertEqual(
                paddle.device.cuda.memory_reserved(), reserved_size
            )
            self.assertEqual(paddle.device.cuda.memory_allocated(), alloc_size)
            del tensor2
            self.assertEqual(
                paddle.device.cuda.memory_reserved(), reserved_size
            )
            self.assertEqual(paddle.device.cuda.memory_allocated(), 0)
            self.assertEqual(
                paddle.device.cuda.max_memory_reserved(), 2 * 1024 * 1024
            )
            self.assertEqual(
                paddle.device.cuda.max_memory_allocated(), 2 * 4 * 256
            )

    def test_memory_stats(self):
        self.func_test_memory_stats()

    def test_reduce_scatter_buffer_uses_vmm(self):
        if not _vmm_runtime_available():
            self.skipTest(
                "Virtual memory allocator is not available on this device."
            )
        params = self._simple_parameters()
        (
            sharding_views,
            buffer_size,
            param_storage,
            grad_storage,
            _,
        ) = build_reduce_scatter_buffer(
            params,
            sharding_degree=1,
            rank=0,
            use_main_grad=False,
            release_grad=True,
        )

        self.assertIsNotNone(param_storage)
        self.assertIsNone(grad_storage)
        self.assertGreater(buffer_size, 0)
        fused_comm_buffer = object.__new__(FusedCommBuffer)
        fused_comm_buffer._param_buffer_meta_tensor = param_storage
        refreshed_meta = fused_comm_buffer.param_buffer_ipc_meta
        self.assertIsNotNone(refreshed_meta)
        self.assertIsInstance(refreshed_meta, tuple)
        self.assertGreater(len(refreshed_meta), 0)
        self.assertEqual(len(sharding_views), len(params))

        values = paddle.arange(param_storage.numel(), dtype=param_storage.dtype)
        values_md5sum = values._md5sum()
        param_storage.set_value(values)
        imported = paddle.base.core.DenseTensor._new_shared_cuda(refreshed_meta)
        imported_tensor = paddle.to_tensor(imported)
        np.testing.assert_allclose(imported_tensor.numpy(), values.numpy())
        del imported_tensor
        self.assertEqual(values._md5sum(), values_md5sum)

    def test_reduce_scatter_meta_refresh_after_tensor_swap(self):
        if not _vmm_runtime_available():
            self.skipTest(
                "Virtual memory allocator is not available on this device."
            )
        params = self._simple_parameters()
        (
            _,
            _,
            param_storage,
            _,
            _,
        ) = build_reduce_scatter_buffer(
            params,
            sharding_degree=1,
            rank=0,
            use_main_grad=False,
            release_grad=True,
        )
        fused_comm_buffer = object.__new__(FusedCommBuffer)
        fused_comm_buffer._param_buffer_meta_tensor = param_storage
        meta_a = fused_comm_buffer.param_buffer_ipc_meta
        imported_a = paddle.to_tensor(
            paddle.base.core.DenseTensor._new_shared_cuda(meta_a)
        )
        np.testing.assert_allclose(
            imported_a.numpy(), param_storage.numpy(), rtol=0, atol=0
        )
        new_storage = paddle.arange(param_storage.numel(), dtype="float32")
        new_storage = new_storage.reshape(param_storage.shape)
        fused_comm_buffer._param_buffer_meta_tensor = new_storage
        meta_b = fused_comm_buffer.param_buffer_ipc_meta
        imported_b = paddle.to_tensor(
            paddle.base.core.DenseTensor._new_shared_cuda(meta_b)
        )
        np.testing.assert_allclose(
            imported_b.numpy(), new_storage.numpy(), rtol=0, atol=0
        )

    def test_fusion_storage_vmm_buffer(self):
        if not _vmm_runtime_available():
            self.skipTest(
                "Virtual memory allocator is not available on this device."
            )
        tensor_a = paddle.zeros([16], dtype="float32")
        tensor_b = paddle.zeros([16], dtype="float32")
        accumulators = {"momentum": {"param_a": tensor_a}}
        master_weights = {"param_b": tensor_b}
        storage = FusionStorage(
            accumulators=accumulators, master_weights=master_weights
        )

        self.assertIsNotNone(storage.buffer_ipc_meta)
        helper = FusionStorageHelper(
            storage.accumulators_meta,
            storage.master_weights_meta,
            storage.merged_model_params_meta,
            storage.buffer_ipc_meta,
        )
        self.assertEqual(storage.buffer._numel(), helper.buffer._numel())
        self.assertGreater(helper.buffer_length, 0)

        helper.buffer.set_value(paddle.full_like(helper.buffer, 3.0))
        np.testing.assert_allclose(
            storage.buffer.numpy(),
            helper.buffer.numpy(),
        )

    def test_multiprocessing_reductions_use_vmm(self):
        if not _vmm_runtime_available():
            self.skipTest(
                "Virtual memory allocator is not available on this device."
            )
        tensor = paddle.arange(0, 64, dtype="float32").reshape([8, 8])
        dense = tensor.value().get_tensor()
        rebuild, meta = reductions._reduce_lodtensor(dense)

        self.assertGreater(len(meta), 1)
        self.assertIs(meta[0], type(dense))

        rebuilt = rebuild(*meta)
        rebuilt_tensor = paddle.to_tensor(rebuilt)
        np.testing.assert_allclose(
            rebuilt_tensor.numpy(),
            tensor.numpy(),
        )


if __name__ == "__main__":
    unittest.main()
