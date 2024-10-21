#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import gradient_checker
import numpy as np
from decorator_helper import prog_scope
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import base
from paddle.base import core


class TestFlipOp_API(unittest.TestCase):
    """Test flip api."""

    def test_static_graph(self):
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(train_program, startup_program):
            axis = [0]
            input = paddle.static.data(
                name='input', dtype='float32', shape=[2, 3]
            )
            output = paddle.flip(input, axis)
            output = paddle.flip(output, -1)
            output = output.flip(0)
            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
            exe = base.Executor(place)
            exe.run(startup_program)
            img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
            res = exe.run(
                train_program, feed={'input': img}, fetch_list=[output]
            )
            out_np = np.array(res[0])
            out_ref = np.array([[3, 2, 1], [6, 5, 4]]).astype(np.float32)
            self.assertTrue(
                (out_np == out_ref).all(),
                msg='flip output is wrong, out =' + str(out_np),
            )

    def test_dygraph(self):
        img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
        with base.dygraph.guard():
            inputs = paddle.to_tensor(img)
            ret = paddle.flip(inputs, [0])
            ret = ret.flip(0)
            ret = paddle.flip(ret, 1)
            out_ref = np.array([[3, 2, 1], [6, 5, 4]]).astype(np.float32)

            self.assertTrue(
                (ret.numpy() == out_ref).all(),
                msg='flip output is wrong, out =' + str(ret.numpy()),
            )


class TestFlipOp(OpTest):
    def setUp(self):
        self.op_type = 'flip'
        self.python_api = paddle.tensor.flip
        self.init_test_case()
        self.init_attrs()
        self.init_dtype()

        if self.is_bfloat16_op():
            self.input = np.random.random(self.in_shape).astype(np.float32)
        else:
            self.input = np.random.random(self.in_shape).astype(self.dtype)

        output = self.calc_ref_res()

        if self.is_bfloat16_op():
            output = output.astype(np.float32)
            self.inputs = {'X': convert_float_to_uint16(self.input)}
            self.outputs = {'Out': convert_float_to_uint16(output)}
        else:
            self.inputs = {'X': self.input.astype(self.dtype)}
            output = output.astype(self.dtype)
            self.outputs = {'Out': output}

    def init_dtype(self):
        self.dtype = np.float64

    def init_attrs(self):
        self.attrs = {"axis": self.axis}

    def test_check_output(self):
        self.check_output(
            check_cinn=True, check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        self.check_grad(["X"], "Out", check_cinn=True, check_pir=True)

    def init_test_case(self):
        self.in_shape = (6, 4, 2, 3)
        self.axis = [0, 1]

    def calc_ref_res(self):
        res = self.input
        if isinstance(self.axis, int):
            return np.flip(res, self.axis)
        for axis in self.axis:
            res = np.flip(res, axis)
        return res


class TestFlipOpAxis1(TestFlipOp):
    def init_test_case(self):
        self.in_shape = (2, 4, 4)
        self.axis = [0]


class TestFlipOpAxis2(TestFlipOp):
    def init_test_case(self):
        self.in_shape = (4, 4, 6, 3)
        self.axis = [0, 2]


class TestFlipOpAxis3(TestFlipOp):
    def init_test_case(self):
        self.in_shape = (4, 3, 1)
        self.axis = [0, 1, 2]


class TestFlipOpAxis4(TestFlipOp):
    def init_test_case(self):
        self.in_shape = (6, 4, 2, 2)
        self.axis = [0, 1, 2, 3]


class TestFlipOpEmptyAxis(TestFlipOp):
    def init_test_case(self):
        self.in_shape = (6, 4, 2, 2)
        self.axis = []


class TestFlipOpNegAxis(TestFlipOp):
    def init_test_case(self):
        self.in_shape = (6, 4, 2, 2)
        self.axis = [-1]


# ----------------flip_fp16----------------
def create_test_fp16_class(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestFlipFP16(parent):
        def init_dtype(self):
            self.dtype = np.float16

        def test_check_output(self):
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
                if core.is_float16_supported(place):
                    self.check_output_with_place(
                        place, check_cinn=True, check_pir=True
                    )

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_grad_with_place(
                    place, ["X"], "Out", check_cinn=True, check_pir=True
                )

    cls_name = "{}_{}".format(parent.__name__, "FP16OP")
    TestFlipFP16.__name__ = cls_name
    globals()[cls_name] = TestFlipFP16


create_test_fp16_class(TestFlipOp)
create_test_fp16_class(TestFlipOpAxis1)
create_test_fp16_class(TestFlipOpAxis2)
create_test_fp16_class(TestFlipOpAxis3)
create_test_fp16_class(TestFlipOpAxis4)
create_test_fp16_class(TestFlipOpEmptyAxis)
create_test_fp16_class(TestFlipOpNegAxis)


# ----------------flip_bf16----------------
def create_test_bf16_class(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda()
        or not core.is_bfloat16_supported(core.CUDAPlace(0)),
        "core is not compiled with CUDA and do not support bfloat16",
    )
    class TestFlipBF16(parent):
        def init_dtype(self):
            self.dtype = np.uint16

        def test_check_output(self):
            place = core.CUDAPlace(0)
            if core.is_bfloat16_supported(place):
                self.check_output_with_place(place, check_pir=True)

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            if core.is_bfloat16_supported(place):
                self.check_grad_with_place(place, ["X"], "Out", check_pir=True)

    cls_name = "{}_{}".format(parent.__name__, "BF16OP")
    TestFlipBF16.__name__ = cls_name
    globals()[cls_name] = TestFlipBF16


create_test_bf16_class(TestFlipOp)
create_test_bf16_class(TestFlipOpAxis1)
create_test_bf16_class(TestFlipOpAxis2)
create_test_bf16_class(TestFlipOpAxis3)
create_test_bf16_class(TestFlipOpAxis4)
create_test_bf16_class(TestFlipOpEmptyAxis)
create_test_bf16_class(TestFlipOpNegAxis)


class TestFlipDoubleGradCheck(unittest.TestCase):
    def flip_wrapper(self, x):
        return paddle.flip(x[0], [0, 1])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not include -1.
        eps = 0.005
        dtype = np.float32

        data = paddle.static.data('data', [3, 2, 2], dtype)
        data.persistable = True
        out = paddle.flip(data, [0, 1])
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.double_grad_check(
            [data], out, x_init=[data_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.flip_wrapper, [data], out, x_init=[data_arr], place=place
        )

    def test_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestFlipTripleGradCheck(unittest.TestCase):
    def flip_wrapper(self, x):
        return paddle.flip(x[0], [0, 1])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not include -1.
        eps = 0.005
        dtype = np.float32

        data = paddle.static.data('data', [3, 2, 2], dtype)
        data.persistable = True
        out = paddle.flip(data, [0, 1])
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.triple_grad_check(
            [data], out, x_init=[data_arr], place=place, eps=eps
        )
        gradient_checker.triple_grad_check_for_dygraph(
            self.flip_wrapper, [data], out, x_init=[data_arr], place=place
        )

    def test_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestFlipError(unittest.TestCase):
    def test_axis(self):
        paddle.enable_static()

        def test_axis_rank():
            input = paddle.static.data(
                name='input', dtype='float32', shape=[2, 3]
            )
            output = paddle.flip(input, axis=[[0]])

        self.assertRaises(TypeError, test_axis_rank)

        def test_axis_rank2():
            input = paddle.static.data(
                name='input', dtype='float32', shape=[2, 3]
            )
            output = paddle.flip(input, axis=[[0, 0], [1, 1]])

        self.assertRaises(TypeError, test_axis_rank2)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
