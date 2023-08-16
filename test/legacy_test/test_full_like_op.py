#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from eager_op_test import OpTest, convert_float_to_uint16

import paddle
import paddle.framework.dtype as dtypes
from paddle.base import core
from paddle.base.framework import convert_np_dtype_to_dtype_
from paddle.static import Program, program_guard


def fill_any_like_wrapper(x, value, out_dtype=None, name=None):
    if isinstance(out_dtype, int):
        tmp_dtype = dtypes.dtype(out_dtype)
    else:
        tmp_dtype = out_dtype
    return paddle.full_like(x, value, tmp_dtype, name)


class TestFullOp(unittest.TestCase):
    """Test fill_any_like op(whose API is full_like) for attr out."""

    def test_attr_tensor_API(self):
        startup_program = Program()
        train_program = Program()
        with program_guard(train_program, startup_program):
            fill_value = 2.0
            input = paddle.static.data(
                name='input', dtype='float32', shape=[2, 3]
            )
            output = paddle.full_like(input, fill_value)
            output_dtype = paddle.full_like(input, fill_value, dtype='float32')

            place = paddle.CPUPlace()
            if core.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
            exe = paddle.static.Executor(place)
            exe.run(startup_program)

            img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)

            res = exe.run(
                train_program, feed={'input': img}, fetch_list=[output]
            )

            out_np = np.array(res[0])
            self.assertTrue(
                not (out_np - np.full_like(img, fill_value)).any(),
                msg="full_like output is wrong, out = " + str(out_np),
            )

    def test_full_like_imperative(self):
        paddle.disable_static()
        input = paddle.arange(6, 10, dtype='float32')
        out = paddle.full_like(input, fill_value=888.88, dtype='float32')
        out_numpy = np.random.random(4).astype("float32")
        out_numpy.fill(888.88)
        self.assertTrue((out.numpy() == out_numpy).all(), True)
        paddle.enable_static()

    def test_full_like_fill_inf(self):
        paddle.disable_static()
        input = paddle.arange(6, 10, dtype='float32')
        out = paddle.full_like(input, fill_value=float('inf'))
        out_numpy = np.random.random(4).astype("float32")
        out_numpy.fill(float('inf'))
        self.assertTrue((out.numpy() == out_numpy).all(), True)
        paddle.enable_static()


class TestFullOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # for ci coverage

            input_data = paddle.static.data(
                name='input', dtype='float32', shape=[2, 3]
            )
            output = paddle.full_like(input_data, 2.0)

            self.assertRaises(
                TypeError,
                paddle.full_like,
                x=input_data,
                fill_value=2,
                dtype='uint4',
            )


class TestFullLikeOp1(OpTest):
    # test basic
    def setUp(self):
        self.op_type = "fill_any_like"
        self.prim_op_type = "comp"
        self.python_api = fill_any_like_wrapper
        self.public_python_api = fill_any_like_wrapper
        self.init_data()
        self.if_enable_cinn()

        bf16_flag = self.dtype == np.uint16
        x = np.zeros(self.shape).astype(np.float32 if bf16_flag else self.dtype)
        x = OpTest.np_dtype_to_base_dtype(x)

        out = np.full_like(x, self.fill_value, self.dtype)

        if bf16_flag:
            x = convert_float_to_uint16(x)
            out = convert_float_to_uint16(out)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}

        self.attrs = {
            'value': self.fill_value,
            'dtype': convert_np_dtype_to_dtype_(self.dtype),
        }

    def init_data(self):
        self.fill_value = 5
        self.shape = [10, 10]
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output(check_prim=True)

    def if_enable_cinn(self):
        pass


class TestFullLikeOp1_ZeroDim(TestFullLikeOp1):
    def init_data(self):
        self.fill_value = 5
        self.shape = []
        self.dtype = np.float32


class TestFullLikeOp2(TestFullLikeOp1):
    def init_data(self):
        self.fill_value = 1000
        self.shape = [10, 10]
        self.dtype = np.float64

    def if_enable_cinn(self):
        pass


class TestFullLikeOp3(TestFullLikeOp1):
    def init_data(self):
        self.fill_value = 8888
        self.shape = [10, 10]
        self.dtype = np.int64

    def if_enable_cinn(self):
        pass


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFullLikeOp4(unittest.TestCase):
    def test_skip_data_transform(self):
        paddle.disable_static()
        x = paddle.to_tensor(
            [1.0, 2.0, 3.0, 4.0], place=paddle.CUDAPinnedPlace()
        )
        out = paddle.full_like(x, 1.0)
        self.assertTrue(
            (out.numpy() == np.ones([4]).astype(np.float32)).all(), True
        )
        paddle.enable_static()


class TestFullLikeFP16Op(TestFullLikeOp1):
    def init_data(self):
        self.fill_value = 6666
        self.shape = [10, 10]
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestFullLikeBF16Op(TestFullLikeOp1):
    def init_data(self):
        self.fill_value = 6666
        self.shape = [10, 10]
        self.dtype = np.uint16


if __name__ == "__main__":
    unittest.main()
