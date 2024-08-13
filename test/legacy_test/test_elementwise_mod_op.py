#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float

import paddle
from paddle import base, static
from paddle.base import core
from paddle.pir_utils import test_with_pir_api


class TestElementwiseModOp(OpTest):
    def init_kernel_type(self):
        self.use_mkldnn = False

    def setUp(self):
        self.op_type = "elementwise_mod"
        self.python_api = paddle.remainder
        self.axis = -1
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, [10, 10]).astype(self.dtype)
        self.out = np.mod(self.x, self.y)

    def init_dtype(self):
        self.dtype = np.int32

    def init_axis(self):
        pass


class TestElementwiseModOp_ZeroDim1(TestElementwiseModOp):
    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, []).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, []).astype(self.dtype)
        self.out = np.mod(self.x, self.y)


class TestElementwiseModOp_ZeroDim2(TestElementwiseModOp):
    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, []).astype(self.dtype)
        self.out = np.mod(self.x, self.y)


class TestElementwiseModOp_ZeroDim3(TestElementwiseModOp):
    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, []).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, [10, 10]).astype(self.dtype)
        self.out = np.mod(self.x, self.y)


class TestElementwiseModOp_scalar(TestElementwiseModOp):
    def init_input_output(self):
        scale_x = random.randint(0, 100000000)
        scale_y = random.randint(1, 100000000)
        self.x = (np.random.rand(2, 3, 4) * scale_x).astype(self.dtype)
        self.y = (np.random.rand(1) * scale_y + 1).astype(self.dtype)
        self.out = np.mod(self.x, self.y)


class TestElementwiseModOpFloat(TestElementwiseModOp):
    def init_dtype(self):
        self.dtype = np.float32

    def init_input_output(self):
        self.x = np.random.uniform(-1000, 1000, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(-100, 100, [10, 10]).astype(self.dtype)
        self.out = np.fmod(self.y + np.fmod(self.x, self.y), self.y)

    def test_check_output(self):
        self.check_output(check_pir=True)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestElementwiseModFP16Op(TestElementwiseModOp):
    def init_dtype(self):
        self.dtype = np.float16

    def init_input_output(self):
        self.x = np.random.uniform(-1000, 1000, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(-100, 100, [10, 10]).astype(self.dtype)
        self.out = np.fmod(self.y + np.fmod(self.x, self.y), self.y)

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestElementwiseModFP16Op_ZeroDim1(TestElementwiseModFP16Op):
    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, []).astype(np.float16)
        self.y = np.random.uniform(0, 1000, []).astype(np.float16)
        self.out = np.fmod(self.y + np.fmod(self.x, self.y), self.y)


class TestElementwiseModFP16Op_ZeroDim2(TestElementwiseModFP16Op):
    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, [10, 10]).astype(np.float16)
        self.y = np.random.uniform(0, 1000, []).astype(np.float16)
        self.out = np.fmod(self.y + np.fmod(self.x, self.y), self.y)


class TestElementwiseModFP16Op_ZeroDim3(TestElementwiseModFP16Op):
    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, []).astype(np.float16)
        self.y = np.random.uniform(0, 1000, [10, 10]).astype(np.float16)
        self.out = np.fmod(self.y + np.fmod(self.x, self.y), self.y)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestElementwiseModBF16Op(OpTest):
    def init_kernel_type(self):
        self.use_mkldnn = False

    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, [10, 10]).astype(np.float32)
        self.x = convert_uint16_to_float(convert_float_to_uint16(self.x))
        self.y = np.random.uniform(0, 1000, [10, 10]).astype(np.float32)
        self.y = convert_uint16_to_float(convert_float_to_uint16(self.y))
        self.out = np.fmod(self.y + np.fmod(self.x, self.y), self.y)

    def setUp(self):
        self.op_type = "elementwise_mod"
        self.python_api = paddle.remainder
        self.public_python_api = paddle.remainder
        self.axis = -1
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()
        self.inputs = {
            'X': convert_float_to_uint16(OpTest.np_dtype_to_base_dtype(self.x)),
            'Y': convert_float_to_uint16(OpTest.np_dtype_to_base_dtype(self.y)),
        }
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': convert_float_to_uint16(self.out)}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def init_dtype(self):
        self.dtype = np.uint16

    def init_axis(self):
        pass


class TestElementwiseModBF16Op_ZeroDim1(TestElementwiseModBF16Op):
    def init_input(self):
        self.x = np.random.uniform(0, 10000, []).astype("float32")
        self.x = convert_uint16_to_float(convert_float_to_uint16(self.x))
        self.y = np.random.uniform(0, 1000, []).astype("float32")
        self.y = convert_uint16_to_float(convert_float_to_uint16(self.y))
        self.out = np.fmod(self.y + np.fmod(self.x, self.y), self.y)


class TestElementwiseModOpDouble(TestElementwiseModOpFloat):
    def init_dtype(self):
        self.dtype = np.float64


class TestRemainderOp(unittest.TestCase):
    def setUp(self):
        self.np_x1 = np.array([2, 3, 8, 7]).astype('int64')
        self.np_y1 = np.array([1, 5, 3, 3]).astype('int64')
        self.z_expected1 = np.array([0, 3, 2, 1])

        self.np_x2 = np.array([-3.3, 11.5, -2, 3.5])
        self.np_y2 = np.array([-1.2, 2.0, 3.3, -2.3])
        self.z_expected2 = np.array([-0.9, 1.5, 1.3, -1.1])

        self.np_x3 = np.array([-3, 11, -2, 3])
        self.np_y3 = np.array([-1, 2, 3, -2])
        self.z_expected3 = np.array([0, 1, 1, -1])

    def _executed_api(self, x, y, name=None):
        return paddle.remainder(x, y, name)

    def test_dygraph(self):
        with base.dygraph.guard():
            x = paddle.to_tensor(self.np_x1)
            y = paddle.to_tensor(self.np_y1)
            z = self._executed_api(x, y)
            np_z = z.numpy()
            self.assertEqual((np_z == self.z_expected1).all(), True)

            x = paddle.to_tensor(self.np_x2)
            y = paddle.to_tensor(self.np_y2)
            z = x % y
            np.testing.assert_allclose(self.z_expected2, z.numpy(), rtol=1e-05)

            x = paddle.to_tensor(self.np_x3, dtype="int64")
            y = paddle.to_tensor(self.np_y3, dtype="int64")
            z = x % y
            np.testing.assert_allclose(self.z_expected3, z.numpy(), rtol=1e-05)

    @test_with_pir_api
    def test_static(self):
        mp, sp = static.Program(), static.Program()
        with static.program_guard(mp, sp):
            x1 = static.data("x1", shape=[4], dtype="int64")
            y1 = static.data("y1", shape=[4], dtype="int64")
            z1 = self._executed_api(x1, y1)

            x2 = static.data("x2", shape=[4], dtype="float64")
            y2 = static.data("y2", shape=[4], dtype="float64")
            z2 = self._executed_api(x2, y2)

            x3 = static.data("x3", shape=[4], dtype="int64")
            y3 = static.data("y3", shape=[4], dtype="int64")
            z3 = self._executed_api(x3, y3)

        exe = static.Executor()
        exe.run(sp)
        [z_np1, z_np2, z_np3] = exe.run(
            mp,
            feed={
                "x1": self.np_x1,
                "y1": self.np_y1,
                "x2": self.np_x2,
                "y2": self.np_y2,
                "x3": self.np_x3,
                "y3": self.np_y3,
            },
            fetch_list=[z1, z2, z3],
        )
        np.testing.assert_allclose(self.z_expected1, z_np1, rtol=1e-05)
        np.testing.assert_allclose(self.z_expected2, z_np2, rtol=1e-05)
        np.testing.assert_allclose(self.z_expected3, z_np3, rtol=1e-05)


class TestRemainderInplaceOp(TestRemainderOp):
    def _executed_api(self, x, y, name=None):
        return x.remainder_(y, name)


class TestRemainderInplaceBroadcastSuccess(unittest.TestCase):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 3, 4).astype('float')
        self.y_numpy = np.random.rand(3, 4).astype('float')

    def test_broadcast_success(self):
        paddle.disable_static()
        self.init_data()
        x = paddle.to_tensor(self.x_numpy)
        y = paddle.to_tensor(self.y_numpy)
        inplace_result = x.remainder_(y)
        numpy_result = self.x_numpy % self.y_numpy
        self.assertEqual((inplace_result.numpy() == numpy_result).all(), True)
        paddle.enable_static()


class TestRemainderInplaceBroadcastSuccess2(
    TestRemainderInplaceBroadcastSuccess
):
    def init_data(self):
        self.x_numpy = np.random.rand(1, 2, 3, 1).astype('float')
        self.y_numpy = np.random.rand(3, 1).astype('float')


class TestRemainderInplaceBroadcastSuccess3(
    TestRemainderInplaceBroadcastSuccess
):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 3, 1, 5).astype('float')
        self.y_numpy = np.random.rand(1, 3, 1, 5).astype('float')


if __name__ == '__main__':
    unittest.main()
