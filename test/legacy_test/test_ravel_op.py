#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core


class TestRavelOp(OpTest):
    def setUp(self):
        self.python_api = paddle.Tensor.ravel
        self.public_python_api = paddle.Tensor.ravel
        self.python_out_sig = ["Out"]
        self.op_type = "flatten_contiguous_range"
        self.prim_op_type = "comp"
        self.start_axis = 0
        self.stop_axis = -1
        self.if_enable_cinn()
        self.init_test_case()
        self.init_test_dtype()
        self.init_input_data()
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
            "XShape": np.random.random(self.in_shape).astype("float32"),
        }

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        if str(self.dtype) in {"float16", "uint16"}:
            self.check_output_with_place(
                core.CUDAPlace(0),
                no_check_set=["XShape"],
                check_prim=True,
                check_pir=True,
                check_prim_pir=True,
            )
        else:
            self.check_output(
                no_check_set=["XShape"],
                check_prim=True,
                check_pir=True,
                check_prim_pir=True,
            )

    def test_check_grad(self):
        if str(self.dtype) in {"float16", "uint16"}:
            self.check_grad_with_place(
                core.CUDAPlace(0),
                ["X"],
                "Out",
                check_prim=True,
                check_pir=True,
            )
        else:
            self.check_grad(["X"], "Out", check_prim=True, check_pir=True)

    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 0
        self.stop_axis = -1
        self.new_shape = 120

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis,
        }

    def init_test_dtype(self):
        self.dtype = "float64"

    def init_input_data(self):
        if str(self.dtype) != "uint16":
            x = np.random.random(self.in_shape).astype(self.dtype)
        else:
            x = np.random.random(self.in_shape).astype("float32")
            x = convert_float_to_uint16(x)

        self.inputs = {"X": x}


class TestRavelFP32Op(TestRavelOp):
    def init_test_dtype(self):
        self.dtype = "float32"


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestRavelFP16Op(TestRavelOp):
    def init_test_dtype(self):
        self.dtype = "float16"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestRavelBF16Op(TestRavelOp):
    def if_enable_cinn(self):
        pass

    def init_test_dtype(self):
        self.dtype = "uint16"


class TestRavelOp_ZeroDim(TestRavelOp):
    def init_test_case(self):
        self.in_shape = ()
        self.start_axis = 0
        self.stop_axis = -1
        self.new_shape = (1,)

    def if_enable_cinn(self):
        self.enable_cinn = False

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis,
        }


class TestRavelFP32Op_ZeroDim(TestRavelOp_ZeroDim):
    def init_test_dtype(self):
        self.dtype = "float32"


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestRavelFP16Op_ZeroDim(TestRavelOp_ZeroDim):
    def init_test_dtype(self):
        self.dtype = "float16"


class TestRavelOpError(unittest.TestCase):
    def test_errors(self):
        image_shape = (2, 3, 4, 4)
        x = (
            np.arange(
                image_shape[0]
                * image_shape[1]
                * image_shape[2]
                * image_shape[3]
            ).reshape(image_shape)
            / 100.0
        )
        x = x.astype('float32')

        def test_InputError():
            out = paddle.Tensor.ravel(x)

        self.assertRaises(ValueError, test_InputError)


class TestStaticRavelPythonAPI(unittest.TestCase):
    def execute_api(self, x):
        return paddle.Tensor.ravel(x)

    def test_static_api(self):
        paddle.enable_static()
        np_x = np.random.rand(2, 3, 4, 4).astype('float32')

        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, paddle.static.Program()):
            x = paddle.static.data(
                name="x", shape=[2, 3, 4, 4], dtype='float32'
            )
            out = self.execute_api(x)

        exe = paddle.static.Executor(place=paddle.CPUPlace())
        fetch_out = exe.run(main_prog, feed={"x": np_x}, fetch_list=[out])
        self.assertTrue((96,) == fetch_out[0].shape)


class TestStaticRavelInferShapePythonAPI(unittest.TestCase):
    def execute_api(self, x):
        return paddle.Tensor.ravel(x)

    def test_static_api(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, paddle.static.Program()):
            x = paddle.static.data(
                name="x", shape=[-1, 3, -1, -1], dtype='float32'
            )
            out = self.execute_api(x)
        self.assertTrue((-1,) == tuple(out.shape))


class TestRavelZeroSizedTensorAPI(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static()
        data = np.random.randn(2, 3, 0)
        x = paddle.to_tensor(data)
        out = paddle.Tensor.ravel(x)
        out_np = data.flatten()
        np.testing.assert_equal(out.numpy(), out_np)

    def test_static(self):
        paddle.enable_static()
        data = np.random.randn(2, 3, 0)
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[2, 3, 0], dtype='float64')
            out = paddle.Tensor.ravel(x)

        exe = paddle.static.Executor(place=paddle.CPUPlace())
        fetch_out = exe.run(main_prog, feed={"x": data}, fetch_list=[out])[0]
        out_np = data.flatten()
        np.testing.assert_equal(fetch_out, out_np)


if __name__ == "__main__":
    unittest.main()
