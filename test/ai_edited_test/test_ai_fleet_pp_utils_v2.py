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

# [AUTO-GENERATED] Tests for paddle/distributed/fleet/meta_parallel/pp_utils/utils.py
# Target: is_float_tensor, get_tensor_dtype, paddle_2_number, number_2_dtype,
#         get_tensor_bytes, _all_gather, tuple_to_dict_helper, dict_to_tuple_helper,
#         convert_tensor_dict_to_tuple, convert_tensor_tuple_to_dict
# Coverage target: ~70.3% -> improved

"""
测试 paddle/distributed/fleet/meta_parallel/pp_utils/utils.py 中的工具函数。

Tests for utility functions in paddle/distributed/fleet/meta_parallel/pp_utils/utils.py.
Covers dtype conversion utilities, tensor byte calculation, _all_gather,
and tensor tuple/dict conversion helpers.
All distributed operations are mocked. Uses actual paddle dtype enums.
"""

import unittest
from unittest.mock import MagicMock, patch

import paddle


class TestFloatTensorChecks(unittest.TestCase):
    """测试浮点张量检查 / Test float tensor checks."""

    def test_is_float_tensor_float32(self):
        """测试 float32 张量 / Test float32 tensor."""
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            is_float_tensor,
        )

        mock_tensor = MagicMock()
        mock_tensor.dtype = paddle.float32
        self.assertTrue(is_float_tensor(mock_tensor))

    def test_is_float_tensor_float16(self):
        """测试 float16 张量 / Test float16 tensor."""
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            is_float_tensor,
        )

        mock_tensor = MagicMock()
        mock_tensor.dtype = paddle.float16
        self.assertTrue(is_float_tensor(mock_tensor))

    def test_is_float_tensor_float64(self):
        """测试 float64 张量 / Test float64 tensor."""
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            is_float_tensor,
        )

        mock_tensor = MagicMock()
        mock_tensor.dtype = paddle.float64
        self.assertTrue(is_float_tensor(mock_tensor))

    def test_is_float_tensor_bfloat16(self):
        """测试 bfloat16 张量 / Test bfloat16 tensor."""
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            is_float_tensor,
        )

        mock_tensor = MagicMock()
        mock_tensor.dtype = paddle.bfloat16
        self.assertTrue(is_float_tensor(mock_tensor))

    def test_is_float_tensor_bool(self):
        """测试 bool 张量 / Test bool tensor."""
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            is_float_tensor,
        )

        mock_tensor = MagicMock()
        mock_tensor.dtype = paddle.bool
        self.assertTrue(is_float_tensor(mock_tensor))

    def test_is_float_tensor_int32(self):
        """测试 int32 张量（不是浮点）/ Test int32 tensor (not float)."""
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            is_float_tensor,
        )

        mock_tensor = MagicMock()
        mock_tensor.dtype = paddle.int32
        self.assertFalse(is_float_tensor(mock_tensor))

    def test_is_float_tensor_int64(self):
        """测试 int64 张量（不是浮点）/ Test int64 tensor (not float)."""
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            is_float_tensor,
        )

        mock_tensor = MagicMock()
        mock_tensor.dtype = paddle.int64
        self.assertFalse(is_float_tensor(mock_tensor))


class TestGetTensorDtype(unittest.TestCase):
    """测试 get_tensor_dtype / Test get_tensor_dtype."""

    def test_float16(self):
        """测试 float16 / Test float16."""
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            get_tensor_dtype,
        )

        result = get_tensor_dtype(paddle.float16)
        self.assertEqual(result, "float16")

    def test_float32(self):
        """测试 float32 / Test float32."""
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            get_tensor_dtype,
        )

        result = get_tensor_dtype(paddle.float32)
        self.assertEqual(result, "float32")

    def test_float64(self):
        """测试 float64 / Test float64."""
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            get_tensor_dtype,
        )

        result = get_tensor_dtype(paddle.float64)
        self.assertEqual(result, "float64")

    def test_bfloat16(self):
        """测试 bfloat16 / Test bfloat16."""
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            get_tensor_dtype,
        )

        result = get_tensor_dtype(paddle.bfloat16)
        self.assertEqual(result, "bfloat16")

    def test_bool(self):
        """测试 bool / Test bool."""
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            get_tensor_dtype,
        )

        result = get_tensor_dtype(paddle.bool)
        self.assertEqual(result, "bool")

    def test_invalid_dtype(self):
        """测试无效 dtype / Test invalid dtype."""
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            get_tensor_dtype,
        )

        with self.assertRaises(AssertionError):
            get_tensor_dtype(paddle.int32)


class TestPaddle2Number(unittest.TestCase):
    """测试 paddle_2_number / Test paddle_2_number."""

    def test_float16(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            paddle_2_number,
        )

        self.assertEqual(paddle_2_number(paddle.float16), 0)

    def test_float32(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            paddle_2_number,
        )

        self.assertEqual(paddle_2_number(paddle.float32), 1)

    def test_float64(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            paddle_2_number,
        )

        self.assertEqual(paddle_2_number(paddle.float64), 2)

    def test_int32(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            paddle_2_number,
        )

        self.assertEqual(paddle_2_number(paddle.int32), 3)

    def test_int64(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            paddle_2_number,
        )

        self.assertEqual(paddle_2_number(paddle.int64), 4)

    def test_bfloat16(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            paddle_2_number,
        )

        self.assertEqual(paddle_2_number(paddle.bfloat16), 5)

    def test_bool(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            paddle_2_number,
        )

        self.assertEqual(paddle_2_number(paddle.bool), 6)

    def test_invalid(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            paddle_2_number,
        )

        with self.assertRaises(AssertionError):
            paddle_2_number("uint8")


class TestNumber2Dtype(unittest.TestCase):
    """测试 number_2_dtype / Test number_2_dtype."""

    def test_0(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            number_2_dtype,
        )

        self.assertEqual(number_2_dtype(0), "float16")

    def test_1(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            number_2_dtype,
        )

        self.assertEqual(number_2_dtype(1), "float32")

    def test_2(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            number_2_dtype,
        )

        self.assertEqual(number_2_dtype(2), "float64")

    def test_3(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            number_2_dtype,
        )

        self.assertEqual(number_2_dtype(3), "int32")

    def test_4(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            number_2_dtype,
        )

        self.assertEqual(number_2_dtype(4), "int64")

    def test_5(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            number_2_dtype,
        )

        self.assertEqual(number_2_dtype(5), "bfloat16")

    def test_6(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            number_2_dtype,
        )

        self.assertEqual(number_2_dtype(6), "bool")

    def test_invalid(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            number_2_dtype,
        )

        with self.assertRaises(AssertionError):
            number_2_dtype(99)


class TestGetTensorBytes(unittest.TestCase):
    """测试 get_tensor_bytes / Test get_tensor_bytes."""

    def test_float32(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            get_tensor_bytes,
        )

        mock_tensor = MagicMock()
        mock_tensor.dtype = paddle.float32
        mock_tensor.numel.return_value = 10
        self.assertEqual(get_tensor_bytes(mock_tensor), 40)

    def test_float64(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            get_tensor_bytes,
        )

        mock_tensor = MagicMock()
        mock_tensor.dtype = paddle.float64
        mock_tensor.numel.return_value = 5
        self.assertEqual(get_tensor_bytes(mock_tensor), 40)

    def test_int64(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            get_tensor_bytes,
        )

        mock_tensor = MagicMock()
        mock_tensor.dtype = paddle.int64
        mock_tensor.numel.return_value = 3
        self.assertEqual(get_tensor_bytes(mock_tensor), 24)

    def test_int32(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            get_tensor_bytes,
        )

        mock_tensor = MagicMock()
        mock_tensor.dtype = paddle.int32
        mock_tensor.numel.return_value = 8
        self.assertEqual(get_tensor_bytes(mock_tensor), 32)

    def test_float16(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            get_tensor_bytes,
        )

        mock_tensor = MagicMock()
        mock_tensor.dtype = paddle.float16
        mock_tensor.numel.return_value = 4
        self.assertEqual(get_tensor_bytes(mock_tensor), 8)

    def test_int8(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            get_tensor_bytes,
        )

        mock_tensor = MagicMock()
        mock_tensor.dtype = paddle.int8
        mock_tensor.numel.return_value = 10
        self.assertEqual(get_tensor_bytes(mock_tensor), 10)

    def test_unknown_dtype(self):
        """测试未知数据类型抛出异常 / Test unknown dtype raises ValueError."""
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            get_tensor_bytes,
        )

        mock_tensor = MagicMock()
        mock_tensor.dtype = "complex128"
        mock_tensor.numel.return_value = 1
        with self.assertRaises(ValueError) as ctx:
            get_tensor_bytes(mock_tensor)
        self.assertIn("unknown data type", str(ctx.exception))

    def test_bfloat16(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            get_tensor_bytes,
        )

        mock_tensor = MagicMock()
        mock_tensor.dtype = paddle.bfloat16
        mock_tensor.numel.return_value = 4
        # bfloat16 is not explicitly handled, should raise
        with self.assertRaises(ValueError):
            get_tensor_bytes(mock_tensor)

    def test_bool_dtype(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            get_tensor_bytes,
        )

        mock_tensor = MagicMock()
        mock_tensor.dtype = paddle.bool
        mock_tensor.numel.return_value = 4
        # bool is not explicitly handled, should raise
        with self.assertRaises(ValueError):
            get_tensor_bytes(mock_tensor)


class TestAllGather(unittest.TestCase):
    """测试 _all_gather / Test _all_gather."""

    def test_non_member_returns_none(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            _all_gather,
        )

        mock_group = MagicMock()
        mock_group.is_member.return_value = False
        result = _all_gather(MagicMock(), group=mock_group)
        self.assertIsNone(result)

    def test_member_with_group(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            _all_gather,
        )

        mock_group = MagicMock()
        mock_group.is_member.return_value = True
        mock_group.id = 3
        mock_group.nranks = 4
        mock_tensor = MagicMock()
        mock_result = MagicMock()
        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.utils._C_ops"
        ) as mock_cops:
            mock_cops.all_gather.return_value = mock_result
            result = _all_gather(
                mock_tensor, group=mock_group, use_calc_stream=True
            )
            mock_cops.all_gather.assert_called_once_with(mock_tensor, 3, 4)
            self.assertEqual(result, mock_result)

    def test_member_no_group(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            _all_gather,
        )

        mock_tensor = MagicMock()
        mock_result = MagicMock()
        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.utils._C_ops"
        ) as mock_cops:
            mock_cops.all_gather.return_value = mock_result
            with patch(
                "paddle.distributed.fleet.meta_parallel.pp_utils.utils.paddle"
            ) as mock_paddle:
                mock_global_group = MagicMock()
                mock_global_group.nranks = 8
                mock_paddle.distributed.collective._get_global_group.return_value = mock_global_group
                result = _all_gather(mock_tensor)
                mock_cops.all_gather.assert_called_once_with(mock_tensor, 0, 8)


class TestTupleDictHelpers(unittest.TestCase):
    """测试 tuple/dict 转换辅助函数 / Test tuple/dict conversion helpers."""

    def test_tuple_to_dict_helper_single_tensor(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            tuple_to_dict_helper,
        )

        mock_tensor = MagicMock()
        mock_tensor.key = "test_key"
        result, use_dict = tuple_to_dict_helper(mock_tensor)
        self.assertTrue(use_dict)

    def test_tuple_to_dict_helper_tuple_without_key(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            tuple_to_dict_helper,
        )

        mock_t1 = MagicMock(spec=[])  # no attributes
        mock_t2 = MagicMock(spec=[])
        result, use_dict = tuple_to_dict_helper((mock_t1, mock_t2))
        self.assertFalse(use_dict)

    def test_dict_to_tuple_helper_dict(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            dict_to_tuple_helper,
        )

        mock_dict = {"key1": MagicMock(), "key2": MagicMock()}
        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.utils.convert_tensor_dict_to_tuple"
        ) as mock_convert:
            mock_convert.return_value = ("t1", "t2")
            result = dict_to_tuple_helper(mock_dict)
            mock_convert.assert_called_once()
            self.assertEqual(result, ("t1", "t2"))

    def test_dict_to_tuple_helper_non_dict(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            dict_to_tuple_helper,
        )

        result = dict_to_tuple_helper(("t1", "t2"))
        self.assertEqual(result, ("t1", "t2"))

    def test_convert_tensor_dict_to_tuple_single(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            convert_tensor_dict_to_tuple,
        )

        mock_tensor = MagicMock()
        mock_dict = {"key1": mock_tensor}
        result = convert_tensor_dict_to_tuple(mock_dict)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertEqual(mock_tensor.key, "key1")

    def test_convert_tensor_dict_to_tuple_list(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            convert_tensor_dict_to_tuple,
        )

        mock_t1 = MagicMock()
        mock_t2 = MagicMock()
        mock_dict = {"key1": [mock_t1, mock_t2]}
        result = convert_tensor_dict_to_tuple(mock_dict)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_convert_tensor_tuple_to_dict_simple(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            convert_tensor_tuple_to_dict,
        )

        mock_t1 = MagicMock()
        mock_t1.key = "key1"
        mock_t2 = MagicMock()
        mock_t2.key = "key2"
        result = convert_tensor_tuple_to_dict((mock_t1, mock_t2))
        self.assertIn("key1", result)
        self.assertIn("key2", result)

    def test_convert_tensor_tuple_to_dict_spaced_key(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            convert_tensor_tuple_to_dict,
        )

        mock_t1 = MagicMock()
        mock_t1.key = "key1 0"
        mock_t2 = MagicMock()
        mock_t2.key = "key1 1"
        result = convert_tensor_tuple_to_dict((mock_t1, mock_t2))
        self.assertIn("key1", result)
        self.assertIsInstance(result["key1"], list)
        self.assertEqual(len(result["key1"]), 2)

    def test_convert_tensor_tuple_to_dict_mixed(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            convert_tensor_tuple_to_dict,
        )

        mock_t1 = MagicMock()
        mock_t1.key = "key1"
        mock_t2 = MagicMock()
        mock_t2.key = "key2 0"
        result = convert_tensor_tuple_to_dict((mock_t1, mock_t2))
        self.assertIn("key1", result)
        self.assertIn("key2", result)


class TestConstants(unittest.TestCase):
    """测试常量定义 / Test constant definitions."""

    def test_float_type_dict_keys(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            FLOAT_TYPE_DICT,
        )

        self.assertEqual(len(FLOAT_TYPE_DICT), 5)
        self.assertEqual(FLOAT_TYPE_DICT[paddle.float16], "float16")
        self.assertEqual(FLOAT_TYPE_DICT[paddle.float32], "float32")
        self.assertEqual(FLOAT_TYPE_DICT[paddle.float64], "float64")
        self.assertEqual(FLOAT_TYPE_DICT[paddle.bfloat16], "bfloat16")
        self.assertEqual(FLOAT_TYPE_DICT[paddle.bool], "bool")

    def test_paddle_to_number_completeness(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            PADDLE_TO_NUMBER,
        )

        self.assertEqual(PADDLE_TO_NUMBER[paddle.float16], 0)
        self.assertEqual(PADDLE_TO_NUMBER[paddle.float32], 1)
        self.assertEqual(PADDLE_TO_NUMBER[paddle.float64], 2)
        self.assertEqual(PADDLE_TO_NUMBER[paddle.int32], 3)
        self.assertEqual(PADDLE_TO_NUMBER[paddle.int64], 4)
        self.assertEqual(PADDLE_TO_NUMBER[paddle.bfloat16], 5)
        self.assertEqual(PADDLE_TO_NUMBER[paddle.bool], 6)

    def test_number_to_dtype_completeness(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            NUMBER_TO_DTYPE,
        )

        self.assertEqual(NUMBER_TO_DTYPE[0], "float16")
        self.assertEqual(NUMBER_TO_DTYPE[1], "float32")
        self.assertEqual(NUMBER_TO_DTYPE[2], "float64")
        self.assertEqual(NUMBER_TO_DTYPE[3], "int32")
        self.assertEqual(NUMBER_TO_DTYPE[4], "int64")
        self.assertEqual(NUMBER_TO_DTYPE[5], "bfloat16")
        self.assertEqual(NUMBER_TO_DTYPE[6], "bool")


if __name__ == '__main__':
    unittest.main()
