# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

sys.path.append("../../legacy_test")
import numpy as np
from op_test import check_out_dtype

sys.path.append("../../legacy_test")
import os

from op_test import OpTest
from test_sum_op import TestReduceOPTensorAxisBase
from utils import dygraph_guard, static_guard

import paddle
from paddle import base
from paddle.base import core


class ApiMaxTest(unittest.TestCase):
    def setUp(self):
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()

    def test_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data = paddle.static.data("data", shape=[10, 10], dtype="float32")
            result_max = paddle.max(x=data, axis=1)
            exe = paddle.static.Executor(self.place)
            input_data = np.random.rand(10, 10).astype(np.float32)
            (res,) = exe.run(feed={"data": input_data}, fetch_list=[result_max])
        self.assertEqual((res == np.max(input_data, axis=1)).all(), True)

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data = paddle.static.data("data", shape=[10, 10], dtype="int64")
            result_max = paddle.max(x=data, axis=0)
            exe = paddle.static.Executor(self.place)
            input_data = np.random.randint(10, size=(10, 10)).astype(np.int64)
            (res,) = exe.run(feed={"data": input_data}, fetch_list=[result_max])
        self.assertEqual((res == np.max(input_data, axis=0)).all(), True)

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data = paddle.static.data("data", shape=[10, 10], dtype="int64")
            result_max = paddle.max(x=data, axis=(0, 1))
            exe = paddle.static.Executor(self.place)
            input_data = np.random.randint(10, size=(10, 10)).astype(np.int64)
            (res,) = exe.run(feed={"data": input_data}, fetch_list=[result_max])
        self.assertEqual((res == np.max(input_data, axis=(0, 1))).all(), True)

    def test_errors(self):
        paddle.enable_static()

        def test_input_type():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                data = np.random.rand(10, 10)
                result_max = paddle.max(x=data, axis=0)

        self.assertRaises(TypeError, test_input_type)

    def test_imperative_api(self):
        paddle.disable_static()
        np_x = np.array([10, 10]).astype('float64')
        x = paddle.to_tensor(np_x)
        z = paddle.max(x, axis=0)
        np_z = z.numpy()
        z_expected = np.array(np.max(np_x, axis=0))
        self.assertEqual((np_z == z_expected).all(), True)

    def test_big_dimension(self):
        paddle.disable_static()
        x = paddle.rand(shape=[2, 2, 2, 2, 2, 2, 2])
        np_x = x.numpy()
        z1 = paddle.max(x, axis=-1)
        z2 = paddle.max(x, axis=6)
        np_z1 = z1.numpy()
        np_z2 = z2.numpy()
        z_expected = np.array(np.max(np_x, axis=6))
        self.assertEqual((np_z1 == z_expected).all(), True)
        self.assertEqual((np_z2 == z_expected).all(), True)

    def test_all_negative_axis(self):
        paddle.disable_static()
        x = paddle.rand(shape=[2, 2])
        np_x = x.numpy()
        z1 = paddle.max(x, axis=(-2, -1))
        np_z1 = z1.numpy()
        z_expected = np.array(np.max(np_x, axis=(0, 1)))
        self.assertEqual((np_z1 == z_expected).all(), True)


class TestOutDtype(unittest.TestCase):
    def test_max(self):
        api_fn = paddle.max
        shape = [10, 16]
        check_out_dtype(
            api_fn,
            in_specs=[(shape,)],
            expect_dtypes=['float32', 'float64', 'int32', 'int64'],
        )


class TestMaxWithTensorAxis1(TestReduceOPTensorAxisBase):
    def init_data(self):
        self.pd_api = paddle.max
        self.np_api = np.max
        self.x = paddle.randn([10, 5, 9, 9], dtype='float64')
        self.np_axis = np.array([1, 2], dtype='int64')
        self.tensor_axis = paddle.to_tensor([1, 2], dtype='int64')


class TestMaxWithTensorAxis2(TestReduceOPTensorAxisBase):
    def init_data(self):
        self.pd_api = paddle.max
        self.np_api = np.max
        self.x = paddle.randn([10, 10, 9, 9], dtype='float64')
        self.np_axis = np.array([0, 1, 2], dtype='int64')
        self.tensor_axis = [
            0,
            paddle.to_tensor([1], 'int64'),
            paddle.to_tensor([2], 'int64'),
        ]


class TestMaxZeroSize1(unittest.TestCase):
    def init_data(self):
        self.shape = [0, 1, 2, 3]
        self.axis = [1, 2, 3]
        self.keepdims = False

    def setUp(self):
        self.init_data()
        self.data = np.random.random(self.shape).astype(np.float64)
        self.expect_res = np.max(
            self.data, axis=tuple(self.axis), keepdims=self.keepdims
        )
        self.places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(core.CUDAPlace(0))

    def test_static(self):
        with static_guard():
            for place in self.places:
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    x = paddle.static.data(
                        "x", shape=self.shape, dtype="float64"
                    )
                    res = paddle.max(x, axis=self.axis, keepdim=self.keepdims)
                    exe = paddle.static.Executor(place)
                    (res,) = exe.run(feed={"x": self.data}, fetch_list=[res])
                np.testing.assert_equal(res, self.expect_res)

    def test_dygraph(self):
        with dygraph_guard():
            x = paddle.to_tensor(self.data)
            res = paddle.max(x, axis=self.axis, keepdim=self.keepdims)
        np.testing.assert_equal(res, self.expect_res)


class TestMaxZeroSize2(TestMaxZeroSize1):
    def init_data(self):
        self.shape = [0, 0, 2]
        self.axis = [2]
        self.keepdims = False


class TestMaxZeroSize3(TestMaxZeroSize1):
    def init_data(self):
        self.shape = [0, 0, 2]
        self.axis = [2]
        self.keepdims = True


class TestMaxOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_max"
        self.python_api = paddle.max
        self.init_data()
        self.prepare_data()

    def init_data(self):
        self.shape = [0, 1, 2]
        self.axis = [1]
        self.dtype = np.float64
        self.keepdims = False

    def prepare_data(self):
        self._input_data = np.random.random(self.shape).astype(self.dtype)
        self._output_data = np.max(
            self._input_data, keepdims=self.keepdims, axis=tuple(self.axis)
        )
        self.inputs = {'X': self._input_data}
        self.outputs = {'Out': self._output_data}
        self.attrs = {"dim": self.axis, "keep_dim": self.keepdims}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            ['Out'],
            check_pir=True,
        )


@unittest.skipIf(
    not core.supports_bfloat16(), "place does not support BF16 evaluation"
)
class TestMaxBfloat16(unittest.TestCase):
    def init_data(self):
        self.shape = [0, 1, 2]
        self.axis = [1]
        self.keepdims = False

    def setUp(self):
        self.init_data()
        data = np.random.random(self.shape).astype(np.float64)
        res = np.max(data, axis=tuple(self.axis), keepdims=self.keepdims)
        self.expect_shape = res.shape

    def test_shape(self):
        with dygraph_guard():
            x = paddle.zeros(self.shape, dtype=paddle.bfloat16)
            res = paddle.max(x, axis=self.axis, keepdim=self.keepdims)
            res = res.numpy()
            np.testing.assert_equal(res.shape, self.expect_shape)


class TestMaxWithNan(unittest.TestCase):
    def _get_places(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if paddle.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        return places

    def _test_with_nan_static(
        self, func, shape, dtype=np.float32, place=paddle.CPUPlace()
    ):
        with (
            static_guard(),
            paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ),
        ):
            x_np = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
            x_np[0, 0] = np.nan
            x = paddle.static.data(name='x', shape=shape, dtype=dtype)
            out = func(x)
            exe = paddle.static.Executor(place)
            res = exe.run(feed={'x': x_np}, fetch_list=[out])
            self.assertTrue(np.isnan(res[0]), "Result should be NaN")

    def _test_with_nan_dynamic(
        self, func, shape, dtype=np.float32, place=paddle.CPUPlace()
    ):
        with dygraph_guard():
            x_np = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
            x_np[0, 0] = np.nan
            x = paddle.to_tensor(x_np, place=place)
            out = func(x)
            self.assertTrue(paddle.isnan(out), "Result should be NaN")

    def test_with_nan(self):
        places = self._get_places()
        for place in places:
            self._test_with_nan_dynamic(paddle.max, (2, 3), place=place)
            self._test_with_nan_static(paddle.max, (2, 3), place=place)


if __name__ == '__main__':
    unittest.main()
