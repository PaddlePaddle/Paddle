#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


class TestFlattenOp(OpTest):
    def setUp(self):
        self.python_api = paddle.flatten
        self.public_python_api = paddle.flatten
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


class TestFlattenFP32Op(TestFlattenOp):
    def init_test_dtype(self):
        self.dtype = "float32"


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestFlattenFP16Op(TestFlattenOp):
    def init_test_dtype(self):
        self.dtype = "float16"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestFlattenBF16Op(TestFlattenOp):
    def if_enable_cinn(self):
        pass

    def init_test_dtype(self):
        self.dtype = "uint16"


class TestFlattenOp_1(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 1
        self.stop_axis = 2
        self.new_shape = (3, 10, 4)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis,
        }


class TestFlattenFP32Op_1(TestFlattenOp_1):
    def init_test_dtype(self):
        self.dtype = "float32"


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestFlattenFP16Op_1(TestFlattenOp_1):
    def init_test_dtype(self):
        self.dtype = "float16"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestFlattenBF16Op_1(TestFlattenOp_1):
    def if_enable_cinn(self):
        pass

    def init_test_dtype(self):
        self.dtype = "uint16"


class TestFlattenOp_2(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 0
        self.stop_axis = 1
        self.new_shape = (6, 5, 4)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis,
        }


class TestFlattenFP32Op_2(TestFlattenOp_2):
    def init_test_dtype(self):
        self.dtype = "float32"


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestFlattenFP16Op_2(TestFlattenOp_2):
    def init_test_dtype(self):
        self.dtype = "float16"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestFlattenBF16Op_2(TestFlattenOp_2):
    def if_enable_cinn(self):
        pass

    def init_test_dtype(self):
        self.dtype = "uint16"


class TestFlattenOp_3(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 0
        self.stop_axis = 2
        self.new_shape = (30, 4)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis,
        }


class TestFlattenFP32Op_3(TestFlattenOp_3):
    def init_test_dtype(self):
        self.dtype = "float32"


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestFlattenFP16Op_3(TestFlattenOp_3):
    def init_test_dtype(self):
        self.dtype = "float16"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestFlattenBF16Op_3(TestFlattenOp_3):
    def if_enable_cinn(self):
        pass

    def init_test_dtype(self):
        self.dtype = "uint16"


class TestFlattenOp_4(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = -2
        self.stop_axis = -1
        self.new_shape = (3, 2, 20)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis,
        }


class TestFlattenFP32Op_4(TestFlattenOp_4):
    def init_test_dtype(self):
        self.dtype = "float32"


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestFlattenFP16Op_4(TestFlattenOp_4):
    def init_test_dtype(self):
        self.dtype = "float16"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestFlattenBF16Op_4(TestFlattenOp_4):
    def if_enable_cinn(self):
        pass

    def init_test_dtype(self):
        self.dtype = "uint16"


class TestFlattenOp_5(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 2
        self.stop_axis = 2
        self.new_shape = (3, 2, 5, 4)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis,
        }


class TestFlattenFP32Op_5(TestFlattenOp_5):
    def init_test_dtype(self):
        self.dtype = "float32"


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestFlattenFP16Op_5(TestFlattenOp_5):
    def init_test_dtype(self):
        self.dtype = "float16"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestFlattenBF16Op_5(TestFlattenOp_5):
    def if_enable_cinn(self):
        pass

    def init_test_dtype(self):
        self.dtype = "uint16"


class TestFlattenOp_ZeroDim(TestFlattenOp):
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


class TestFlattenFP32Op_ZeroDim(TestFlattenOp_ZeroDim):
    def init_test_dtype(self):
        self.dtype = "float32"


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestFlattenFP16Op_ZeroDim(TestFlattenOp_ZeroDim):
    def init_test_dtype(self):
        self.dtype = "float16"


class TestFlattenOpSixDims(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 3, 2, 4, 4)
        self.start_axis = 3
        self.stop_axis = 5
        self.new_shape = (3, 2, 3, 32)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis,
        }


class TestFlattenFP32OpSixDims(TestFlattenOpSixDims):
    def init_test_dtype(self):
        self.dtype = "float32"


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestFlattenFP16OpSixDims(TestFlattenOpSixDims):
    def init_test_dtype(self):
        self.dtype = "float16"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestFlattenBF16OpSixDims(TestFlattenOpSixDims):
    def if_enable_cinn(self):
        pass

    def init_test_dtype(self):
        self.dtype = "uint16"


class TestFlatten2OpError(unittest.TestCase):
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

        def test_ValueError1():
            x_var = paddle.static.data(
                name="x", shape=image_shape, dtype='float32'
            )
            out = paddle.flatten(x_var, start_axis=3, stop_axis=1)

        self.assertRaises(ValueError, test_ValueError1)

        def test_ValueError2():
            x_var = paddle.static.data(
                name="x", shape=image_shape, dtype='float32'
            )
            paddle.flatten(x_var, start_axis=10, stop_axis=1)

        self.assertRaises(ValueError, test_ValueError2)

        def test_ValueError3():
            x_var = paddle.static.data(
                name="x", shape=image_shape, dtype='float32'
            )
            paddle.flatten(x_var, start_axis=2, stop_axis=10)

        self.assertRaises(ValueError, test_ValueError3)

        def test_ValueError4():
            x_var = paddle.static.data(
                name="x", shape=image_shape, dtype='float32'
            )
            paddle.flatten(x_var, start_axis=2.0, stop_axis=10)

        self.assertRaises(ValueError, test_ValueError4)

        def test_ValueError5():
            x_var = paddle.static.data(
                name="x", shape=image_shape, dtype='float32'
            )
            paddle.flatten(x_var, start_axis=2, stop_axis=10.0)

        self.assertRaises(ValueError, test_ValueError5)

        def test_InputError():
            out = paddle.flatten(x)

        self.assertRaises(ValueError, test_InputError)


class TestStaticFlattenPythonAPI(unittest.TestCase):
    def execute_api(self, x, start_axis=0, stop_axis=-1):
        return paddle.flatten(x, start_axis, stop_axis)

    def test_static_api(self):
        paddle.enable_static()
        np_x = np.random.rand(2, 3, 4, 4).astype('float32')

        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, paddle.static.Program()):
            x = paddle.static.data(
                name="x", shape=[2, 3, 4, 4], dtype='float32'
            )
            out = self.execute_api(x, start_axis=-2, stop_axis=-1)

        exe = paddle.static.Executor(place=paddle.CPUPlace())
        fetch_out = exe.run(main_prog, feed={"x": np_x}, fetch_list=[out])
        self.assertTrue((2, 3, 16) == fetch_out[0].shape)


class TestStaticFlattenInferShapePythonAPI(unittest.TestCase):
    def execute_api(self, x, start_axis=0, stop_axis=-1):
        return paddle.flatten(x, start_axis, stop_axis)

    def test_static_api(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, paddle.static.Program()):
            x = paddle.static.data(
                name="x", shape=[-1, 3, -1, -1], dtype='float32'
            )
            out = self.execute_api(x, start_axis=2, stop_axis=3)
        self.assertTrue((-1, 3, -1) == tuple(out.shape))


class TestStaticInplaceFlattenPythonAPI(TestStaticFlattenPythonAPI):
    def execute_api(self, x, start_axis=0, stop_axis=-1):
        return x.flatten_(start_axis, stop_axis)


class TestFlattenPython(unittest.TestCase):
    def test_python_api(self):
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
            out = paddle.flatten(x)

        self.assertRaises(ValueError, test_InputError)

        def test_Negative():
            paddle.disable_static()
            img = paddle.to_tensor(x)
            out = paddle.flatten(img, start_axis=-2, stop_axis=-1)
            return out.numpy().shape

        res_shape = test_Negative()
        self.assertTrue((2, 3, 16) == res_shape)


class TestDygraphInplaceFlattenPython(unittest.TestCase):
    def test_python_api(self):
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

        def test_Negative():
            paddle.disable_static()
            img = paddle.to_tensor(x)
            out = img.flatten_(start_axis=-2, stop_axis=-1)
            return out.numpy().shape

        res_shape = test_Negative()
        self.assertTrue((2, 3, 16) == res_shape)
        paddle.enable_static()


class TestFlatten0DTensorOpError(unittest.TestCase):
    def test_errors(self):
        image_shape = ()
        x = np.random.uniform(-1.0, 1.0, []).astype('float32')

        def test_ValueError1():
            x_var = paddle.static.data(
                name="x", shape=image_shape, dtype='float32'
            )
            out = paddle.flatten(x_var, start_axis=10, stop_axis=0)

        self.assertRaises(ValueError, test_ValueError1)

        def test_ValueError2():
            x_var = paddle.static.data(
                name="x", shape=image_shape, dtype='float32'
            )
            out = paddle.flatten(x_var, start_axis=0, stop_axis=10)

        self.assertRaises(ValueError, test_ValueError2)


class TestFlattenZeroSizedTensorAPI(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static()
        data = np.random.randn(2, 3, 0)
        x = paddle.to_tensor(data)
        out = paddle.flatten(x)
        out_np = data.flatten()
        np.testing.assert_equal(out.numpy(), out_np)

    def test_static(self):
        paddle.enable_static()
        data = np.random.randn(2, 3, 0)
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[2, 3, 0], dtype='float64')
            out = paddle.flatten(x)

        exe = paddle.static.Executor(place=paddle.CPUPlace())
        fetch_out = exe.run(main_prog, feed={"x": data}, fetch_list=[out])[0]
        out_np = data.flatten()
        np.testing.assert_equal(fetch_out, out_np)


if __name__ == "__main__":
    unittest.main()
