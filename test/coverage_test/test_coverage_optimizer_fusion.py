# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED]
# Target file: python/paddle/optimizer/fusion_utils.py
# Coverage target: get_current_device_type, get_align, FusionStorage class
# 未覆盖行: FusionStorageHelper class (requires IPC metadata)

import unittest

import numpy as np

import paddle
from paddle.optimizer.fusion_utils import (
    FusionStorage,
    get_align,
    get_current_device_type,
)


class TestGetCurrentDeviceType(unittest.TestCase):
    """Test get_current_device_type function.
    测试 get_current_device_type 函数。"""

    def test_returns_string(self):
        """get_current_device_type should return a string.
        get_current_device_type 应该返回字符串。"""
        result = get_current_device_type()
        self.assertIsInstance(result, str)

    def test_returns_valid_device(self):
        """get_current_device_type should return 'gpu', 'xpu', or 'unknown'.
        get_current_device_type 应该返回 'gpu'、'xpu' 或 'unknown'。"""
        result = get_current_device_type()
        self.assertIn(result, ["gpu", "xpu", "unknown"])

    def test_consistent_result(self):
        """Multiple calls should return the same result (cached).
        多次调用应返回相同结果（缓存）。"""
        result1 = get_current_device_type()
        result2 = get_current_device_type()
        self.assertEqual(result1, result2)


class TestGetAlign(unittest.TestCase):
    """Test get_align function.
    测试 get_align 函数。"""

    def setUp(self):
        paddle.disable_static()

    def test_get_align_float32(self):
        """Test alignment for float32 tensor.
        测试 float32 张量的对齐。"""
        t = paddle.zeros([10], dtype=paddle.float32)
        # Should return an integer alignment value
        result = get_align(t)
        self.assertIsInstance(result, (int, np.integer))

    def test_get_align_float16(self):
        """Test alignment for float16 tensor.
        测试 float16 张量的对齐。"""
        t = paddle.zeros([10], dtype=paddle.float16)
        result = get_align(t)
        self.assertIsInstance(result, (int, np.integer))

    def test_get_align_bfloat16(self):
        """Test alignment for bfloat16 tensor.
        测试 bfloat16 张量的对齐。"""
        t = paddle.zeros([10], dtype=paddle.bfloat16)
        result = get_align(t)
        self.assertIsInstance(result, (int, np.integer))

    def test_get_align_large_tensor(self):
        """Test alignment for a large tensor that may already be aligned.
        测试已经对齐的大张量的对齐值。"""
        # 256 bytes / 4 bytes per float32 = 64 elements -> should be aligned
        t = paddle.zeros([64], dtype=paddle.float32)
        result = get_align(t)
        self.assertEqual(result, 0)

    def test_get_align_unaligned_tensor(self):
        """Test alignment for a small unaligned tensor.
        测试小的未对齐张量的对齐值。"""
        t = paddle.zeros([3], dtype=paddle.float32)
        result = get_align(t)
        # 3 * 4 = 12 bytes, 256 - 12 = 244 remaining, 244 / 4 = 61
        self.assertGreaterEqual(result, 0)

    def test_get_align_2d_tensor(self):
        """Test alignment for 2D tensor.
        测试二维张量的对齐值。"""
        t = paddle.zeros([5, 10], dtype=paddle.float32)
        result = get_align(t)
        self.assertIsInstance(result, (int, np.integer))

    def test_get_align_result_non_negative(self):
        """Alignment result should always be non-negative.
        对齐结果应始终为非负。"""
        t = paddle.zeros([7], dtype=paddle.float32)
        result = get_align(t)
        self.assertGreaterEqual(result, 0)


class TestFusionStorage(unittest.TestCase):
    """Test FusionStorage class.
    测试 FusionStorage 类。"""

    def setUp(self):
        paddle.disable_static()

    def test_basic_creation(self):
        """Test basic FusionStorage creation with accumulators and master_weights.
        测试使用 accumulators 和 master_weights 创建基本 FusionStorage。"""
        acc = paddle.randn([10], dtype=paddle.float32)
        mw = paddle.randn([10], dtype=paddle.float32)
        accumulators = {"momentum": {"weight": acc}}
        master_weights = {"weight": mw}
        storage = FusionStorage(accumulators, master_weights)
        self.assertIsNotNone(storage.buffer)
        self.assertEqual(storage.buffer.dtype, paddle.float32)

    def test_with_merged_model_params(self):
        """Test FusionStorage with merged_model_params.
        测试带有 merged_model_params 的 FusionStorage。"""
        acc = paddle.randn([8], dtype=paddle.float32)
        mw = paddle.randn([8], dtype=paddle.float32)
        mp = paddle.randn([8], dtype=paddle.float32)
        accumulators = {"momentum": {"w": acc}}
        master_weights = {"w": mw}
        merged_model_params = {"w": mp}
        storage = FusionStorage(
            accumulators,
            master_weights,
            merged_model_params=merged_model_params,
        )
        self.assertIsNotNone(storage.buffer)
        self.assertIsNotNone(storage.merged_model_params_meta)

    def test_buffer_shape(self):
        """Test that buffer size accounts for alignment padding.
        测试 buffer 大小考虑了对齐填充。"""
        acc = paddle.randn([10], dtype=paddle.float32)
        mw = paddle.randn([10], dtype=paddle.float32)
        accumulators = {"momentum": {"w": acc}}
        master_weights = {"w": mw}
        storage = FusionStorage(accumulators, master_weights)
        # Buffer should be at least as large as raw data
        raw_size = 10 + 10
        self.assertGreaterEqual(storage.buffer.shape[0], raw_size)

    def test_mapping_tensor_preserves_values(self):
        """Test that mapping_tensor copies values into the buffer correctly.
        测试 mapping_tensor 正确地将值复制到 buffer 中。"""
        acc_val = paddle.ones([5], dtype=paddle.float32) * 3.14
        mw_val = paddle.ones([5], dtype=paddle.float32) * 2.71
        accumulators = {"momentum": {"w": acc_val.clone()}}
        master_weights = {"w": mw_val.clone()}
        storage = FusionStorage(accumulators, master_weights)
        # Check that accumulator values are in buffer
        acc_meta = storage.accumulators_meta["momentum"]["w"]
        buf_slice = storage.buffer._slice(acc_meta["start"], acc_meta["end"])
        # The first 5 elements should be 3.14
        np.testing.assert_array_almost_equal(
            buf_slice.numpy()[:5], np.full(5, 3.14), decimal=5
        )
        # Check that master weight values are in buffer
        mw_meta = storage.master_weights_meta["w"]
        mw_slice = storage.buffer._slice(mw_meta["start"], mw_meta["end"])
        np.testing.assert_array_almost_equal(
            mw_slice.numpy()[:5], np.full(5, 2.71), decimal=5
        )

    def test_accumulators_meta_structure(self):
        """Test that accumulators_meta has correct keys.
        测试 accumulators_meta 包含正确的键。"""
        acc = paddle.randn([4], dtype=paddle.float32)
        mw = paddle.randn([4], dtype=paddle.float32)
        accumulators = {"sgd": {"param1": acc}}
        master_weights = {"param1": mw}
        storage = FusionStorage(accumulators, master_weights)
        self.assertIn("sgd", storage.accumulators_meta)
        self.assertIn("param1", storage.accumulators_meta["sgd"])
        meta = storage.accumulators_meta["sgd"]["param1"]
        self.assertIn("start", meta)
        self.assertIn("end", meta)
        self.assertIn("name", meta)
        self.assertIn("shape", meta)

    def test_master_weights_meta_structure(self):
        """Test that master_weights_meta has correct keys.
        测试 master_weights_meta 包含正确的键。"""
        acc = paddle.randn([6], dtype=paddle.float32)
        mw = paddle.randn([6], dtype=paddle.float32)
        accumulators = {"momentum": {"p": acc}}
        master_weights = {"p": mw}
        storage = FusionStorage(accumulators, master_weights)
        self.assertIn("p", storage.master_weights_meta)
        meta = storage.master_weights_meta["p"]
        self.assertIn("start", meta)
        self.assertIn("end", meta)
        self.assertIn("name", meta)
        self.assertIn("shape", meta)

    def test_merged_model_params_meta_structure(self):
        """Test that merged_model_params_meta is populated correctly.
        测试 merged_model_params_meta 正确填充。"""
        acc = paddle.randn([4], dtype=paddle.float32)
        mw = paddle.randn([4], dtype=paddle.float32)
        mp = paddle.randn([4], dtype=paddle.float32)
        accumulators = {"adam": {"p": acc}}
        master_weights = {"p": mw}
        merged = {"p": mp}
        storage = FusionStorage(
            accumulators, master_weights, merged_model_params=merged
        )
        self.assertIn("p", storage.merged_model_params_meta)
        meta = storage.merged_model_params_meta["p"]
        self.assertIn("start", meta)
        self.assertIn("end", meta)
        self.assertIn("shape", meta)

    def test_multiple_accumulator_groups(self):
        """Test with multiple accumulator groups.
        测试多个累加器组的情况。"""
        acc1 = paddle.randn([3], dtype=paddle.float32)
        acc2 = paddle.randn([3], dtype=paddle.float32)
        mw = paddle.randn([3], dtype=paddle.float32)
        accumulators = {
            "momentum": {"w": acc1},
            "variance": {"w": acc2},
        }
        master_weights = {"w": mw}
        storage = FusionStorage(accumulators, master_weights)
        self.assertIn("momentum", storage.accumulators_meta)
        self.assertIn("variance", storage.accumulators_meta)

    def test_none_merged_model_params(self):
        """Test FusionStorage with merged_model_params=None (default).
        测试 merged_model_params=None（默认值）的 FusionStorage。"""
        acc = paddle.randn([4], dtype=paddle.float32)
        mw = paddle.randn([4], dtype=paddle.float32)
        accumulators = {"momentum": {"w": acc}}
        master_weights = {"w": mw}
        storage = FusionStorage(
            accumulators, master_weights, merged_model_params=None
        )
        self.assertIsNone(storage.merged_model_params)
        self.assertEqual(storage.merged_model_params_meta, {})

    def test_assert_accumulators_dict(self):
        """Test that accumulators must be a dict.
        测试 accumulators 必须是字典。"""
        mw = paddle.randn([4], dtype=paddle.float32)
        with self.assertRaises(AssertionError):
            FusionStorage("not_a_dict", {"w": mw})

    def test_assert_master_weights_dict(self):
        """Test that master_weights must be a dict.
        测试 master_weights 必须是字典。"""
        acc = paddle.randn([4], dtype=paddle.float32)
        with self.assertRaises(AssertionError):
            FusionStorage({"w": acc}, "not_a_dict")

    def test_assert_merged_model_params_type(self):
        """Test that merged_model_params must be dict or None.
        测试 merged_model_params 必须是字典或 None。"""
        acc = paddle.randn([4], dtype=paddle.float32)
        mw = paddle.randn([4], dtype=paddle.float32)
        with self.assertRaises(AssertionError):
            FusionStorage({"w": acc}, {"w": mw}, merged_model_params="invalid")

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "GPU required for IPC metadata"
    )
    def test_buffer_ipc_meta_on_gpu(self):
        """Test buffer_ipc_meta property on GPU.
        测试 GPU 上的 buffer_ipc_meta 属性。"""
        acc = paddle.randn([4], dtype=paddle.float32)
        mw = paddle.randn([4], dtype=paddle.float32)
        accumulators = {"momentum": {"w": acc.cuda()}}
        master_weights = {"w": mw.cuda()}
        storage = FusionStorage(accumulators, master_weights)
        # On CUDA (non-ROCm), buffer_ipc_meta should return IPC metadata
        meta = storage.buffer_ipc_meta
        # meta could be None on ROCm or could be a tuple on CUDA
        if not paddle.is_compiled_with_rocm():
            self.assertIsNotNone(meta)

    def test_dtype_float16_storage(self):
        """Test FusionStorage with float16 dtype.
        测试 float16 类型的 FusionStorage。"""
        acc = paddle.randn([8], dtype=paddle.float16)
        mw = paddle.randn([8], dtype=paddle.float16)
        # Note: FusionStorage requires matching dtype, but we test float16
        # which uses align=2 vs float32 align=4
        accumulators = {"momentum": {"w": acc}}
        master_weights = {"w": mw}
        storage = FusionStorage(
            accumulators, master_weights, dtype=paddle.float16
        )
        self.assertEqual(storage.dtype, paddle.float16)
        self.assertEqual(storage.buffer.dtype, paddle.float16)

    def test_meta_shape_matches(self):
        """Test that stored shape in metadata matches original tensor shape.
        测试元数据中存储的形状与原始张量形状匹配。"""
        acc = paddle.randn([3, 4], dtype=paddle.float32)
        mw = paddle.randn([3, 4], dtype=paddle.float32)
        accumulators = {"momentum": {"w": acc}}
        master_weights = {"w": mw}
        storage = FusionStorage(accumulators, master_weights)
        self.assertEqual(
            tuple(storage.accumulators_meta["momentum"]["w"]["shape"]),
            (3, 4),
        )
        self.assertEqual(
            tuple(storage.master_weights_meta["w"]["shape"]), (3, 4)
        )


if __name__ == "__main__":
    unittest.main()
