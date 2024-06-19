# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from op_test import convert_float_to_uint16, convert_uint16_to_float

import paddle
from paddle.base import core
from paddle.base.variable_index import _setitem_static
from paddle.pir_utils import test_with_pir_api


class TestSetitemInDygraph(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.ndtype = np.float64
        self.dtype = 'float64'

    def test_advanced_index(self):
        np_data = np.zeros((3, 4, 5, 6), dtype='float32').astype(self.ndtype)
        if self.dtype == 'bfloat16':
            np_data = convert_uint16_to_float(convert_float_to_uint16(np_data))
        if self.dtype == 'complex64' or self.dtype == 'complex128':
            np_data = np_data + 1j * np_data

        x = paddle.to_tensor(np_data, dtype=self.dtype)
        np_data[[0, 1], [1, 2], [1]] = 10.0
        x[[0, 1], [1, 2], [1]] = 10.0

        if self.dtype == 'bfloat16':
            x = paddle.cast(x, dtype='float32')
        np.testing.assert_allclose(x.numpy(), np_data)

    def test_combined_index_1(self):
        np_data = np.zeros((3, 4, 5, 6), dtype='float32').astype(self.ndtype)
        if self.dtype == 'bfloat16':
            np_data = convert_uint16_to_float(convert_float_to_uint16(np_data))
        if self.dtype == 'complex64' or self.dtype == 'complex128':
            np_data = np_data + 1j * np_data

        x = paddle.to_tensor(np_data, dtype=self.dtype)
        np_data[[0, 1], :, [1, 2]] = 10.0
        x[[0, 1], :, [1, 2]] = 10.0

        if self.dtype == 'bfloat16':
            x = paddle.cast(x, dtype='float32')
        np.testing.assert_allclose(x.numpy(), np_data)

    def test_combined_index_2(self):
        np_data = np.ones((3, 4, 5, 6), dtype='float32').astype(self.ndtype)
        if self.dtype == 'bfloat16':
            np_data = convert_uint16_to_float(convert_float_to_uint16(np_data))
        if self.dtype == 'complex64' or self.dtype == 'complex128':
            np_data = np_data + 1j * np_data
        x = paddle.to_tensor(np_data, dtype=self.dtype)

        np_data[:, 1, [1, 2], 0] = 10.0
        x[:, 1, [1, 2], 0] = 10.0

        if self.dtype == 'bfloat16':
            x = paddle.cast(x, dtype='float32')
        np.testing.assert_allclose(x.numpy(), np_data)

    def test_combined_index_3(self):
        np_data = np.ones((3, 4, 5, 6), dtype='int32').astype(self.ndtype)
        if self.dtype == 'bfloat16':
            np_data = convert_uint16_to_float(convert_float_to_uint16(np_data))
        if self.dtype == 'complex64' or self.dtype == 'complex128':
            np_data = np_data + 1j * np_data
        x = paddle.to_tensor(np_data, dtype=self.dtype)

        np_data[:, [True, False, True, False], [1, 4]] = 10
        x[:, [True, False, True, False], [1, 4]] = 10

        if self.dtype == 'bfloat16':
            x = paddle.cast(x, dtype='float32')
        np.testing.assert_allclose(x.numpy(), np_data)

    def test_index_has_range(self):
        np_data = np.ones((3, 4, 5, 6), dtype='int32').astype(self.ndtype)
        if self.dtype == 'bfloat16':
            np_data = convert_uint16_to_float(convert_float_to_uint16(np_data))
        if self.dtype == 'complex64' or self.dtype == 'complex128':
            np_data = np_data + 1j * np_data
        x = paddle.to_tensor(np_data, dtype=self.dtype)

        np_data[:, range(3), [1, 2, 4]] = 10
        x[:, range(3), [1, 2, 4]] = 10

        if self.dtype == 'bfloat16':
            x = paddle.cast(x, dtype='float32')
        np.testing.assert_allclose(x.numpy(), np_data)

    def test_src_value_with_different_dtype_1(self):
        # basic-indexing, with set_value op
        np_data = np.ones((3, 4, 5, 6), dtype='int32').astype(self.ndtype)
        if self.dtype == 'bfloat16':
            np_data = convert_uint16_to_float(convert_float_to_uint16(np_data))
        if self.dtype == 'complex64' or self.dtype == 'complex128':
            np_data = np_data + 1j * np_data
        np_value = np.zeros((6,), dtype='float32')
        x = paddle.to_tensor(np_data, dtype=self.dtype)
        v = paddle.to_tensor(np_value, dtype=self.dtype)

        np_data[0, 2, 3] = np_value
        x[0, 2, 3] = v

        if self.dtype == 'bfloat16':
            x = paddle.cast(x, dtype='float32')
        np.testing.assert_allclose(x.numpy(), np_data)

    def test_src_value_with_different_dtype_2(self):
        # combined-indexing, with index_put op
        np_data = np.ones((3, 4, 5, 6), dtype='float32').astype(self.ndtype)
        if self.dtype == 'bfloat16':
            np_data = convert_uint16_to_float(convert_float_to_uint16(np_data))
        if self.dtype == 'complex64' or self.dtype == 'complex128':
            np_data = np_data + 1j * np_data
        np_value = np.zeros((6,), dtype='int64')

        x = paddle.to_tensor(np_data, dtype=self.dtype)
        v = paddle.to_tensor(np_value, dtype=self.dtype)

        np_data[:, [1, 0], 3] = np_value
        x[:, [1, 0], 3] = v

        if self.dtype == 'bfloat16':
            x = paddle.cast(x, dtype='float32')
        np.testing.assert_allclose(x.numpy(), np_data)

    def test_indexing_with_bool_list1(self):
        # test bool-list indexing when axes num less than x.rank
        np_data = (
            np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6)).astype(self.ndtype)
        )
        if self.dtype == 'bfloat16':
            np_data = convert_uint16_to_float(convert_float_to_uint16(np_data))
        if self.dtype == 'complex64' or self.dtype == 'complex128':
            np_data = np_data + 1j * np_data
        x = paddle.to_tensor(np_data, dtype=self.dtype)

        np_data[[True, False, True], [False, False, False, True]] = 7
        x[[True, False, True], [False, False, False, True]] = 7

        if self.dtype == 'bfloat16':
            x = paddle.cast(x, dtype='float32')

        np.testing.assert_allclose(x.numpy(), np_data, verbose=True)

    def test_indexing_with_bool_list2(self):
        # test bool-list indexing when axes num less than x.rank
        np_data = (
            np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6)).astype(self.ndtype)
        )
        if self.dtype == 'bfloat16':
            np_data = convert_uint16_to_float(convert_float_to_uint16(np_data))
        if self.dtype == 'complex64' or self.dtype == 'complex128':
            np_data = np_data + 1j * np_data
        x = paddle.to_tensor(np_data, dtype=self.dtype)

        np_data[
            [True, False, True],
            [False, False, True, False],
            [True, False, False, True, False],
        ] = 8

        x[
            [True, False, True],
            [False, False, True, False],
            [True, False, False, True, False],
        ] = 8

        if self.dtype == 'bfloat16':
            x = paddle.cast(x, dtype='float32')
        np.testing.assert_allclose(x.numpy(), np_data)

    def test_indexing_is_multi_dim_list(self):
        # indexing is multi-dim int list, should be treat as one index, like numpy>=1.23
        np_data = (
            np.arange(3 * 4 * 5 * 6).reshape((6, 5, 4, 3)).astype(self.ndtype)
        )
        if self.dtype == 'bfloat16':
            np_data = convert_uint16_to_float(convert_float_to_uint16(np_data))
        if self.dtype == 'complex64' or self.dtype == 'complex128':
            np_data = np_data + 1j * np_data
        x = paddle.to_tensor(np_data, dtype=self.dtype)
        np_data[np.array([[2, 3, 4], [1, 2, 5]])] = 100
        x[[[2, 3, 4], [1, 2, 5]]] = 100

        if self.dtype == 'bfloat16':
            x = paddle.cast(x, dtype='float32')
        np.testing.assert_allclose(x.numpy(), np_data)

    def test_indexing_is_boolean_true(self):
        # indexing is boolean, should improve rank of tensor and then treat it as advanced indexing.
        np_data = (
            np.arange(3 * 4 * 5 * 6).reshape((6, 5, 4, 3)).astype(self.ndtype)
        )

        if self.dtype == 'bfloat16':
            np_data = convert_uint16_to_float(convert_float_to_uint16(np_data))
        if self.dtype == 'complex64' or self.dtype == 'complex128':
            np_data = np_data + 1j * np_data

        x = paddle.to_tensor(np_data, dtype=self.dtype)
        np_data[2, True, :, 1] = 100
        x[2, True, :, 1] = 100

        if self.dtype == 'bfloat16':
            x = paddle.cast(x, dtype='float32')

        np.testing.assert_allclose(x.numpy(), np_data)

    def test_indexing_is_boolean_false(self):
        # indexing is boolean, should improve rank of tensor and then treat it as advanced indexing.
        np_data = (
            np.arange(3 * 4 * 5 * 6).reshape((6, 5, 4, 3)).astype(self.ndtype)
        )

        if self.dtype == 'bfloat16':
            np_data = convert_uint16_to_float(convert_float_to_uint16(np_data))
        if self.dtype == 'complex64' or self.dtype == 'complex128':
            np_data = np_data + 1j * np_data

        x = paddle.to_tensor(np_data, dtype=self.dtype)
        np_data[False] = 100
        x[False] = 100

        if self.dtype == 'bfloat16':
            x = paddle.cast(x, dtype='float32')

        np.testing.assert_allclose(x.numpy(), np_data)

    def test_combined_indexing_and_value_is_tensor_1(self):
        # value is tensor with same shape to getitem and index will be adjusted
        np_data = np.ones((3, 3)).astype(self.ndtype)
        value_data = np.array([-1, -1, -1]).astype(self.ndtype)

        if self.dtype == 'bfloat16':
            np_data = convert_uint16_to_float(convert_float_to_uint16(np_data))
            value_data = convert_uint16_to_float(
                convert_float_to_uint16(value_data)
            )
        if self.dtype == 'complex64' or self.dtype == 'complex128':
            np_data = np_data + 1j * np_data
            value_data = value_data + 1j * value_data

        x = paddle.to_tensor(np_data, dtype=self.dtype)
        v = paddle.to_tensor(value_data, dtype=self.dtype)

        np_data[:, [0, 2]] = np_data[:, [0, 2]] + np.expand_dims(value_data, -1)
        x[:, [0, 2]] = x[:, [0, 2]] + v.unsqueeze(-1)

        if self.dtype == 'bfloat16':
            x = paddle.cast(x, dtype='float32')

        np.testing.assert_allclose(x.numpy(), np_data)

    def test_combined_indexing_and_value_is_tensor_2(self):
        # value is tensor needed to broadcast and index will be adjusted
        np_data = np.ones((3, 4, 5, 6)).astype(self.ndtype)
        value_data = np.arange(3 * 4 * 2 * 1).reshape((3, 4, 2, 1))

        if self.dtype == 'bfloat16':
            np_data = convert_uint16_to_float(convert_float_to_uint16(np_data))
            value_data = convert_uint16_to_float(
                convert_float_to_uint16(value_data)
            )
        if self.dtype == 'complex64' or self.dtype == 'complex128':
            np_data = np_data + 1j * np_data
            value_data = value_data + 1j * value_data

        x = paddle.to_tensor(np_data, dtype=self.dtype)
        v = paddle.to_tensor(value_data, dtype=self.dtype)
        x[..., [1, 4], ::2] = v

        np_data[..., [1, 4], ::2] = value_data
        if self.dtype == 'bfloat16':
            x = paddle.cast(x, dtype='float32')
        np.testing.assert_allclose(x.numpy(), np_data)

    def test_combined_indexing_and_value_is_tensor_3(self):
        # value is tensor and index will be adjusted
        # and the value rank is less than original tensor
        np_data = np.ones((3, 4, 5, 6)).astype(self.ndtype)
        value_data = np.arange(2 * 3 * 5).reshape((2, 3, 5))

        if self.dtype == 'bfloat16':
            np_data = convert_uint16_to_float(convert_float_to_uint16(np_data))
            value_data = convert_uint16_to_float(
                convert_float_to_uint16(value_data)
            )
        if self.dtype == 'complex64' or self.dtype == 'complex128':
            np_data = np_data + 1j * np_data
            value_data = value_data + 1j * value_data

        x = paddle.to_tensor(np_data, dtype=self.dtype)
        v = paddle.to_tensor(value_data, dtype=self.dtype)
        x[:, [1, 3], :, [3, 4]] = v

        np_data[:, [1, 3], :, [3, 4]] = value_data

        if self.dtype == 'bfloat16':
            x = paddle.cast(x, dtype='float32')
        np.testing.assert_allclose(x.numpy(), np_data)

    def test_inplace_with_stride(self):
        np_v = np.random.randn(3, 1).astype(self.ndtype)
        if self.dtype == 'bfloat16':
            np_v = convert_uint16_to_float(convert_float_to_uint16(np_v))
        if self.dtype == 'complex64' or self.dtype == 'complex128':
            np_v = np_v + 1j * np_v
        v = paddle.to_tensor(np_v, dtype=self.dtype)
        v.stop_gradient = False
        vv = v

        zero = paddle.randn((3, 3, 5))
        zero.stop_gradient = False

        zero1 = zero * 1
        zero1[1, paddle.to_tensor([2, 0, 1])] = vv

        loss = zero1.sum()
        loss.backward()

        expected_v_grad = np.ones((3, 1)) * 5.0
        if self.dtype == 'bfloat16':
            np.testing.assert_allclose(
                v.grad.cast('float32').numpy(), expected_v_grad
            )
        elif self.dtype == 'bool':
            np.testing.assert_equal(
                v.grad.numpy(), expected_v_grad.astype('bool')
            )
        else:
            np.testing.assert_equal(v.grad.numpy(), expected_v_grad)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestFP16SetitemInDygraph(TestSetitemInDygraph):
    def setUp(self):
        paddle.disable_static()
        self.ndtype = np.float16
        self.dtype = 'float16'


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestBF16SetitemInDygraph(TestSetitemInDygraph):
    def setUp(self):
        paddle.disable_static()
        self.ndtype = np.float32
        self.dtype = 'bfloat16'


class TestFP32SetitemInDygraph(TestSetitemInDygraph):
    def setUp(self):
        paddle.disable_static()
        self.ndtype = np.float32
        self.dtype = 'float32'


class TestUINT8SetitemInDygraph(TestSetitemInDygraph):
    def setUp(self):
        paddle.disable_static()
        self.ndtype = np.uint8
        self.dtype = 'uint8'


class TestINT8SetitemInDygraph(TestSetitemInDygraph):
    def setUp(self):
        paddle.disable_static()
        self.ndtype = np.int8
        self.dtype = 'int8'


class TestINT16SetitemInDygraph(TestSetitemInDygraph):
    def setUp(self):
        paddle.disable_static()
        self.ndtype = np.int16
        self.dtype = 'int16'


class TestINT32SetitemInDygraph(TestSetitemInDygraph):
    def setUp(self):
        paddle.disable_static()
        self.ndtype = np.int32
        self.dtype = 'int32'


class TestINT64SetitemInDygraph(TestSetitemInDygraph):
    def setUp(self):
        paddle.disable_static()
        self.ndtype = np.int64
        self.dtype = 'int64'


class TestBOOLSetitemInDygraph(TestSetitemInDygraph):
    def setUp(self):
        paddle.disable_static()
        self.ndtype = np.bool_
        self.dtype = 'bool'


class TestComplex64SetitemInDygraph(TestSetitemInDygraph):
    def setUp(self):
        paddle.disable_static()
        self.ndtype = np.float32
        self.dtype = 'complex64'


class TestComplex128SetitemInDygraph(TestSetitemInDygraph):
    def setUp(self):
        paddle.disable_static()
        self.ndtype = np.float64
        self.dtype = 'complex128'


class TestSetitemInStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.exe = paddle.static.Executor()

    @test_with_pir_api
    def test_advanced_index(self):
        # multi-int tensor
        np_data = np.zeros((3, 4, 5, 6), dtype='float32')
        np_data[[0, 1], [1, 2], [1]] = 10.0
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.zeros((3, 4, 5, 6), dtype='float32')
            y = _setitem_static(x, ([0, 1], [1, 2], [1]), 10.0)
            res = self.exe.run(fetch_list=[y])

        np.testing.assert_allclose(res[0], np_data)

    @test_with_pir_api
    def test_combined_index_1(self):
        # int tensor + slice (without decreasing axes)
        np_data = np.zeros((3, 4, 5, 6), dtype='float32')
        np_data[[0, 1], :, [1, 2]] = 10.0
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.zeros((3, 4, 5, 6), dtype='float32')
            y = _setitem_static(
                x, ([0, 1], slice(None, None, None), [1, 2]), 10.0
            )
            res = self.exe.run(fetch_list=[y])

        np.testing.assert_allclose(res[0], np_data)

    @test_with_pir_api
    def test_combined_index_2(self):
        # int tensor + slice (with decreasing axes)
        np_data = np.ones((3, 4, 5, 6), dtype='float32')
        np_data[:, 1, [1, 2], 0] = 10.0
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.ones((3, 4, 5, 6), dtype='float32')
            y = _setitem_static(
                x, (slice(None, None, None), 1, [1, 2], 0), 10.0
            )
            res = self.exe.run(fetch_list=[y])

        np.testing.assert_allclose(res[0], np_data)

    @test_with_pir_api
    def test_combined_index_3(self):
        # int tensor + bool tensor + slice (without decreasing axes)
        np_data = np.ones((3, 4, 5, 6), dtype='int32')
        np_data[:, [True, False, True, False], [1, 4]] = 10
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.ones((3, 4, 5, 6), dtype='int32')
            y = _setitem_static(
                x,
                (slice(None, None, None), [True, False, True, False], [1, 4]),
                10,
            )
            res = self.exe.run(fetch_list=[y])

        np.testing.assert_allclose(res[0], np_data)

    @test_with_pir_api
    def test_combined_index_4(self):
        # int tensor (with ranks > 1) + bool tensor + slice (with decreasing axes)
        np_data = np.ones((3, 4, 5, 6), dtype='int32')
        np_data[[0, 0], [True, False, True, False], [[0, 2], [1, 4]], 4] = 16
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.ones((3, 4, 5, 6), dtype='int32')
            y = _setitem_static(
                x,
                ([0, 0], [True, False, True, False], [[0, 2], [1, 4]], 4),
                16,
            )
            res = self.exe.run(fetch_list=[y])

        np.testing.assert_allclose(res[0], np_data)

    @test_with_pir_api
    def test_combined_index_5(self):
        # int tensor + slice + Ellipsis
        np_data = np.ones((3, 4, 5, 6), dtype='int32')
        np_data[..., [1, 4, 3], ::2] = 5
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.ones((3, 4, 5, 6), dtype='int32')
            y = _setitem_static(
                x,
                (..., [1, 4, 3], slice(None, None, 2)),
                5,
            )
            res = self.exe.run(fetch_list=[y])

        np.testing.assert_allclose(res[0], np_data)

    @test_with_pir_api
    def test_index_has_range(self):
        np_data = np.ones((3, 4, 5, 6), dtype='int32')
        np_data[:, range(3), [1, 2, 4]] = 10
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.ones((3, 4, 5, 6), dtype='int32')
            y = _setitem_static(
                x,
                (slice(None, None), range(3), [1, 2, 4]),
                10,
            )
            res = self.exe.run(fetch_list=[y])

        np.testing.assert_allclose(res[0], np_data)

    @test_with_pir_api
    def test_src_value_with_different_dtype_1(self):
        # basic-indexing, with set_value op
        np_data = np.ones((3, 4, 5, 6), dtype='int32')
        np_value = np.zeros((6,), dtype='float32')
        np_data[0, 2, 3] = np_value

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.ones((3, 4, 5, 6), dtype='int32')
            v = paddle.zeros((6,), dtype='float32')
            y = _setitem_static(
                x,
                (0, 2, 3),
                v,
            )
            res = self.exe.run(fetch_list=[y])

        np.testing.assert_allclose(res[0], np_data)

    @test_with_pir_api
    def test_src_value_with_different_dtype_2(self):
        # combined-indexing, with index_put op
        np_data = np.ones((3, 4, 5, 6), dtype='float32')
        np_value = np.zeros((6,), dtype='int64')
        np_data[:, [1, 0], 3] = np_value

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.ones((3, 4, 5, 6), dtype='float32')
            v = paddle.zeros((6,), dtype='int64')
            y = _setitem_static(
                x,
                (slice(None, None), [1, 0], 3),
                v,
            )
            res = self.exe.run(fetch_list=[y])

        np.testing.assert_allclose(res[0], np_data)

    @test_with_pir_api
    def test_indexing_with_bool_list1(self):
        # test bool-list indexing when axes num less than x.rank
        np_data = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
        np_data[[True, False, True], [False, False, False, True]] = 7

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
            y = _setitem_static(
                x, ([True, False, True], [False, False, False, True]), 7
            )
            res = self.exe.run(fetch_list=[y])

        np.testing.assert_allclose(res[0], np_data)

    @test_with_pir_api
    def test_indexing_with_bool_list2(self):
        # test bool-list indexing when axes num less than x.rank
        np_data = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
        np_data[
            [True, False, True],
            [False, False, True, False],
            [True, False, False, True, False],
        ] = 8
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
            y = _setitem_static(
                x,
                (
                    [True, False, True],
                    [False, False, True, False],
                    [True, False, False, True, False],
                ),
                8,
            )
            res = self.exe.run(fetch_list=[y])

        np.testing.assert_allclose(res[0], np_data)

    @test_with_pir_api
    def test_indexing_is_multi_dim_list(self):
        # indexing is multi-dim int list, should be treat as one index, like numpy>=1.23
        np_data = np.arange(3 * 4 * 5 * 6).reshape((6, 5, 4, 3))
        np_data[np.array([[2, 3, 4], [1, 2, 5]])] = 10
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.arange(3 * 4 * 5 * 6).reshape((6, 5, 4, 3))
            y = _setitem_static(x, [[[2, 3, 4], [1, 2, 5]]], 10)

            res = self.exe.run(fetch_list=[y])

        np.testing.assert_allclose(res[0], np_data)

    @test_with_pir_api
    def test_indexing_is_boolean_true(self):
        # indexing is boolean, should improve rank of tensor and then treat it as advanced indexing.
        np_data = np.arange(3 * 4 * 5 * 6).reshape((6, 5, 4, 3))
        np_data[2, True, :, 1] = 100

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.arange(3 * 4 * 5 * 6).reshape((6, 5, 4, 3))
            y = _setitem_static(x, (2, True, slice(None), 1), 100)

            res = self.exe.run(fetch_list=[y])

        np.testing.assert_allclose(res[0], np_data)

    @test_with_pir_api
    def test_indexing_is_boolean_false(self):
        # indexing is boolean, should improve rank of tensor and then treat it as advanced indexing.
        np_data = np.arange(3 * 4 * 5 * 6).reshape((6, 5, 4, 3))
        np_data[False] = 100

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.arange(3 * 4 * 5 * 6).reshape((6, 5, 4, 3))
            y = _setitem_static(x, False, 100)

            res = self.exe.run(fetch_list=[y])

        np.testing.assert_allclose(res[0], np_data)

    @test_with_pir_api
    def test_combined_indexing_and_value_is_tensor_1(self):
        # value is tensor with same shape to getitem and index will be adjusted
        np_data = np.ones((3, 3), dtype='int32')
        value_data = np.array([-1, -1, -1])
        np_data[:, [0, 2]] = np_data[:, [0, 2]] * np.expand_dims(value_data, -1)
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.ones((3, 3), dtype='int32')
            v = paddle.to_tensor([-1, -1, -1], dtype='int32')
            y = _setitem_static(
                x,
                (slice(None), [0, 2]),
                x[:, [0, 2]] * v.unsqueeze(-1),
            )
            res = self.exe.run(fetch_list=[y])

        np.testing.assert_allclose(res[0], np_data)

    @test_with_pir_api
    def test_combined_indexing_and_value_is_tensor_2(self):
        # value is tensor needed to broadcast and index will be adjusted
        np_data = np.ones((3, 4, 5, 6), dtype='int32')
        value_data = np.arange(3 * 4 * 2 * 1).reshape((3, 4, 2, 1))
        np_data[..., [1, 4], ::2] = value_data

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.ones((3, 4, 5, 6), dtype='int32')
            v = paddle.arange(3 * 4 * 2 * 1).reshape((3, 4, 2, 1))

            y = _setitem_static(
                x,
                (..., [1, 4], slice(None, None, 2)),
                v,
            )

            res = self.exe.run(fetch_list=[y])

        np.testing.assert_allclose(res[0], np_data)

    @test_with_pir_api
    def test_combined_indexing_and_value_is_tensor_3(self):
        # value is tensor and index will be adjusted
        # and the value rank is less than original tensor
        np_data = np.ones((3, 4, 5, 6), dtype='int32')
        value_data = np.arange(2 * 3 * 5).reshape((2, 3, 5))
        np_data[:, [1, 3], :, [3, 4]] = value_data

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.ones((3, 4, 5, 6), dtype='int32')
            v = paddle.arange(2 * 3 * 5).reshape((2, 3, 5))
            y = _setitem_static(
                x,
                (slice(None), [1, 3], slice(None), [3, 4]),
                v,
            )

            res = self.exe.run(fetch_list=[y])

        np.testing.assert_allclose(res[0], np_data)


if __name__ == '__main__':
    unittest.main()
