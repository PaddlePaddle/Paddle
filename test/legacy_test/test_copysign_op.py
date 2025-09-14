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
from op_test import OpTest, convert_float_to_uint16, get_device_place

import paddle
from paddle.base import core

np.random.seed(100)
paddle.seed(100)


def ref_copysign(x, y):
    out_dtype = x.dtype
    return np.copysign(x, y).astype(out_dtype)


def ref_grad_copysign(x, y, dout):
    out = np.copysign(x, y)
    return dout * out / x


class TestCopySignOp(OpTest):
    def setUp(self):
        self.op_type = "copysign"
        self.python_api = paddle.copysign
        self.init_config()
        self.inputs = {'x': self.x, 'y': self.y}
        self.target = ref_copysign(self.inputs['x'], self.inputs['y'])
        self.outputs = {'out': self.target}

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(['x', 'y'], ['out'], check_pir=True)

    def test_check_grad_ignore_x(self):
        self.check_grad(['y'], ['out'], check_pir=True)

    def test_check_grad_ignore_y(self):
        self.check_grad(['x'], ['out'], check_pir=True)

    def init_config(self):
        self.x = np.random.randn(20, 6).astype('float64')
        self.y = np.random.randn(20, 6).astype('float64')


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestCopySignBF16(OpTest):
    def setUp(self):
        self.op_type = "copysign"
        self.python_api = paddle.copysign
        self.init_dtype()
        np.random.seed(1024)
        x = np.random.randn(20, 6).astype(np.float32)
        y = np.random.randn(20, 6).astype(np.float32)
        out = ref_copysign(x, y)
        self.inputs = {
            'x': convert_float_to_uint16(x),
            'y': convert_float_to_uint16(y),
        }
        self.outputs = {'out': convert_float_to_uint16(out)}
        self.place = core.CUDAPlace(0)

    def init_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place, check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ['x', 'y'], ['out'], check_pir=True
        )

    def test_check_grad_ignore_x(self):
        self.check_grad_with_place(
            self.place, ['y'], ['out'], no_grad_set=set('x'), check_pir=True
        )

    def test_check_grad_ignore_y(self):
        self.check_grad_with_place(
            self.place, ['x'], ['out'], no_grad_set=set('y'), check_pir=True
        )


class TestCopySignAPI(unittest.TestCase):
    def setUp(self):
        self.input_init()
        self.place_init()

    def input_init(self):
        self.x = np.random.randn(20, 6).astype('float64')
        self.y = np.random.randn(20, 6).astype('float64')

    def place_init(self):
        self.place = get_device_place()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(
                name='x', shape=self.x.shape, dtype=self.x.dtype
            )
            if isinstance(self.y, (float, int)):
                y = self.y
            else:
                y = paddle.static.data(
                    name='y', shape=self.y.shape, dtype=self.y.dtype
                )
            out = paddle.copysign(x, y)
            exe = paddle.static.Executor(self.place)
            if isinstance(self.y, (float, int)):
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x": self.x},
                    fetch_list=[out],
                )
            else:
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x": self.x, "y": self.y},
                    fetch_list=[out],
                )

            out_ref = ref_copysign(self.x, self.y)
            np.testing.assert_allclose(out_ref, res[0])
            out_ref_dtype = out_ref.dtype
            np.testing.assert_equal((out_ref_dtype == res[0].dtype), True)
        paddle.disable_static()

    def test_dygraph_api(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        out = paddle.copysign(x, y)
        out_ref = ref_copysign(self.x, self.y)
        np.testing.assert_allclose(out_ref, out.numpy())
        out_ref_dtype = out_ref.dtype
        np.testing.assert_equal((out_ref_dtype == out.numpy().dtype), True)
        paddle.enable_static()


class TestCopySignBool(TestCopySignAPI):
    def input_init(self):
        dtype = np.bool_
        self.x = (np.random.randn(10, 20) * 10).astype(dtype)
        self.y = (np.random.randn(10, 20) * 10).astype(dtype)


class TestCopySignUint8(TestCopySignAPI):
    def input_init(self):
        dtype = np.uint8
        self.x = (np.random.randn(10, 20) * 10).astype(dtype)
        self.y = (np.random.randn(10, 20) * 10).astype(dtype)


class TestCopySignInt8(TestCopySignAPI):
    def input_init(self):
        dtype = np.int8
        self.x = (np.random.randn(10, 20) * 10).astype(dtype)
        self.y = (np.random.randn(10, 20) * 10).astype(dtype)


class TestCopySignInt16(TestCopySignAPI):
    def input_init(self):
        dtype = np.int16
        self.x = (np.random.randn(10, 20) * 10).astype(dtype)
        self.y = (np.random.randn(10, 20) * 10).astype(dtype)


class TestCopySignInt32(TestCopySignAPI):
    def input_init(self):
        dtype = np.int32
        self.x = (np.random.randn(10, 20) * 10).astype(dtype)
        self.y = (np.random.randn(10, 20) * 10).astype(dtype)


class TestCopySignInt64(TestCopySignAPI):
    def input_init(self):
        dtype = np.int64
        self.x = (np.random.randn(10, 20) * 10).astype(dtype)
        self.y = (np.random.randn(10, 20) * 10).astype(dtype)


class TestCopySignFloat16(TestCopySignAPI):
    def input_init(self):
        dtype = np.float16
        self.x = (np.random.randn(10, 20) * 10).astype(dtype)
        self.y = (np.random.randn(10, 20) * 10).astype(dtype)


class TestCopySignFloat32(TestCopySignAPI):
    def input_init(self):
        dtype = np.float32
        self.x = (np.random.randn(10, 20) * 10).astype(dtype)
        self.y = (np.random.randn(10, 20) * 10).astype(dtype)


class TestCopySignFloat64(TestCopySignAPI):
    def input_init(self):
        dtype = np.float64
        self.x = (np.random.randn(10, 20) * 10).astype(dtype)
        self.y = (np.random.randn(10, 20) * 10).astype(dtype)


class TestCopySignNumberY(TestCopySignAPI):
    def input_init(self):
        dtype = np.float32
        self.x = (np.random.randn(10, 20) * 10).astype(dtype)
        self.y = -2.0


class TestCopySignZeroCase1(TestCopySignAPI):
    def input_init(self):
        self.x = np.zeros(shape=(10, 20))
        self.y = np.zeros(shape=(10, 20))


class TestCopySignZeroCase2(TestCopySignAPI):
    def input_init(self):
        self.x = np.zeros(shape=(10, 20))
        self.y = np.random.randn(10, 20)


class TestCopySignZeroCase3(TestCopySignAPI):
    def input_init(self):
        self.x = np.random.randn(10, 20)
        self.y = np.zeros(shape=(10, 20))


class TestCopySignZeroDimCase1(TestCopySignAPI):
    def input_init(self):
        self.x = np.random.randn(0, 0)
        self.y = np.random.randn(0, 0)


class TestCopySignZeroDimCase2(TestCopySignAPI):
    def input_init(self):
        self.x = np.random.randn(0, 5, 10)
        self.y = np.random.randn(0, 5, 10)


class TestCopySignSpecialZeroCase1(TestCopySignAPI):
    def input_init(self):
        self.x = np.array([1, 2, 3]).astype(np.float32)
        self.y = np.array([0, +0, -0]).astype(np.float32)


class TestCopySignSpecialZeroCase2(TestCopySignAPI):
    def input_init(self):
        self.x = np.array([0, +0, -0]).astype(np.float32)
        self.y = np.array([1, 2, 3]).astype(np.float32)


class TestCopySignBroadcastCase1(TestCopySignAPI):
    def input_init(self):
        dtype = np.float16
        self.x = (np.random.randn(3, 4, 5) * 10).astype(dtype)
        self.y = (np.random.randn(5) * 10).astype(dtype)


class TestCopySignBroadcastCase2(TestCopySignAPI):
    def input_init(self):
        dtype = np.float16
        self.x = (np.random.randn(3, 4, 5) * 10).astype(dtype)
        self.y = (np.random.randn(4, 5) * 10).astype(dtype)


class TestCopySignBroadcastCase3(TestCopySignAPI):
    def input_init(self):
        dtype = np.float16
        self.x = (np.random.randn(4, 5) * 10).astype(dtype)
        self.y = (np.random.randn(3, 4, 5) * 10).astype(dtype)


class TestCopySignZeroSize1(TestCopySignAPI):
    def input_init(self):
        self.x = np.random.randn(0, 5)
        self.y = np.random.randn(0, 5)

    def place_init(self):
        self.place = paddle.CPUPlace()


class TestCopySignZeroSize2(TestCopySignAPI):
    def input_init(self):
        self.x = np.random.randn(0, 5)
        self.y = np.random.randn(3, 0, 5)

    def place_init(self):
        self.place = paddle.CPUPlace()


class TestCopySignZeroSize3(TestCopySignAPI):
    def input_init(self):
        self.x = np.random.randn(3, 0, 5)
        self.y = np.random.randn(0, 5)


class TestCopySignZeroSize4(TestCopySignAPI):
    def input_init(self):
        self.x = np.random.randn(5, 0)
        self.y = np.random.randn(3, 5, 0)


class TestCopySignZeroSize5(TestCopySignAPI):
    def input_init(self):
        self.x = np.random.randn(2, 5)
        self.y = np.random.randn(0, 2, 5)


class TestCopySignZeroSize6(TestCopySignAPI):
    def input_init(self):
        self.x = np.random.randn(0)
        self.y = np.random.randn(0)


class TestCopySignTypePromotion(TestCopySignAPI):
    def input_init(self):
        self.x = np.random.randn(2, 5).astype(np.float64)
        self.y = np.random.randn(2, 5).astype(np.float32)


class TestCopySignNan1(TestCopySignAPI):
    def input_init(self):
        self.x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        self.y = np.array(
            [[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float64
        )
        self.y.view('uint64')[0, 0] |= np.uint64(0x8000000000000000)


class TestCopySignNan2(TestCopySignAPI):
    def input_init(self):
        self.x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        self.y = np.array(
            [[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float64
        )
        self.y.view('uint64')[0, 0] &= ~np.uint64(0x8000000000000000)


class TestCopySignNan3(TestCopySignAPI):
    def input_init(self):
        self.x = np.array([[np.nan, 2.0], [3.0, 4.0]])
        self.y = np.array([[-1.0, np.nan], [np.nan, np.nan]])


class TestCopySignNan4(TestCopySignAPI):
    def input_init(self):
        self.x = np.array([[np.nan, 2.0], [3.0, 4.0]])
        self.y = np.array([[-0.0, np.nan], [np.nan, np.nan]])


class TestCopySignNan5(TestCopySignAPI):
    def input_init(self):
        self.x = np.array([[np.nan, 2.0], [3.0, 4.0]])
        self.y = np.array([[0.0, np.nan], [np.nan, np.nan]])


class TestCopySignNan6(TestCopySignAPI):
    def input_init(self):
        self.x = np.array([[np.nan, 2.0], [3.0, 4.0]])
        self.y = np.array([[1.0, np.nan], [np.nan, np.nan]])


class TestCopySignNan7(TestCopySignAPI):
    def input_init(self):
        self.x = np.array([[np.nan, 2.0], [3.0, 4.0]], dtype=np.float64)
        self.y = np.array(
            [[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float64
        )
        self.y.view('uint64')[0, 0] |= np.uint64(0x8000000000000000)


class TestCopySignNan8(TestCopySignAPI):
    def input_init(self):
        self.x = np.array([[np.nan, 2.0], [3.0, 4.0]], dtype=np.float64)
        self.y = np.array(
            [[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float64
        )
        self.y.view('uint64')[0, 0] &= ~np.uint64(0x8000000000000000)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestCopySignOp_Stride(OpTest):
    no_need_check_grad = True

    def setUp(self):
        self.op_type = "copysign"
        self.python_api = paddle.copysign
        self.public_python_api = paddle.copysign
        self.transpose_api = paddle.transpose
        self.as_stride_api = paddle.as_strided
        self.init_dtype()
        self.init_input_output()

        self.inputs_stride = {
            'x': OpTest.np_dtype_to_base_dtype(self.x),
            'y': OpTest.np_dtype_to_base_dtype(self.y_trans),
        }

        self.inputs = {
            'x': OpTest.np_dtype_to_base_dtype(self.x),
            'y': OpTest.np_dtype_to_base_dtype(self.y),
        }

        self.outputs = {'out': self.out}

    def init_dtype(self):
        self.dtype = np.float64
        self.val_dtype = np.float64

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_strided_forward = True
        self.check_output(
            place,
        )

    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = ref_copysign(self.x, self.y)
        self.perm = [1, 0]
        self.y_trans = np.transpose(self.y, self.perm)

    def test_check_gradient(self):
        pass


class TestCopySignOp_Stride1(TestCopySignOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(self.dtype)
        self.out = ref_copysign(self.x, self.y)
        self.perm = [0, 1, 3, 2]
        self.y_trans = np.transpose(self.y, self.perm)


class TestCopySignOp_Stride2(TestCopySignOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(self.dtype)
        self.out = ref_copysign(self.x, self.y)
        self.perm = [0, 2, 1, 3]
        self.y_trans = np.transpose(self.y, self.perm)


class TestCopySignOp_Stride3(TestCopySignOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [20, 2, 13, 1]).astype(self.dtype)
        self.out = ref_copysign(self.x, self.y)
        self.perm = [0, 1, 3, 2]
        self.y_trans = np.transpose(self.y, self.perm)


class TestCopySignOp_Stride4(TestCopySignOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, [1, 2, 13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [20, 2, 13, 1]).astype(self.dtype)
        self.out = ref_copysign(self.x, self.y)
        self.perm = [1, 0, 2, 3]
        self.y_trans = np.transpose(self.y, self.perm)


class TestCopySignOp_Stride5(TestCopySignOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "as_stride"
        self.x = np.random.uniform(0.1, 1, [23, 10, 1, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [23, 2, 13, 20]).astype(self.dtype)
        self.y_trans = self.y
        self.y = self.y[:, 0:1, :, 0:1]
        self.out = ref_copysign(self.x, self.y)
        self.shape_param = [23, 1, 13, 1]
        self.stride_param = [520, 260, 20, 1]


class TestCopySignOp_Stride_ZeroDim1(TestCopySignOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, []).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = ref_copysign(self.x, self.y)
        self.perm = [1, 0]
        self.y_trans = np.transpose(self.y, self.perm)


class TestCopySignOp_Stride_ZeroSize1(TestCopySignOp_Stride):
    def init_data(self):
        self.strided_input_type = "transpose"
        self.x = np.random.rand(1, 0, 2).astype('float32')
        self.y = np.random.rand(3, 0, 1).astype('float32')
        self.out = ref_copysign(self.x, self.y)
        self.perm = [2, 1, 0]
        self.y_trans = np.transpose(self.y, self.perm)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
