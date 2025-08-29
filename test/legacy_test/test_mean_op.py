#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import gradient_checker
import numpy as np
from decorator_helper import prog_scope
from op_test import OpTest, OpTestTool, convert_float_to_uint16, get_places
from test_sum_op import TestReduceOPTensorAxisBase

import paddle
from paddle import base
from paddle.base import Program, core, program_guard

np.random.seed(10)


def mean_wrapper(x, axis=None, keepdim=False, reduce_all=False):
    if reduce_all:
        return paddle.mean(x, list(range(len(x.shape))), keepdim)
    return paddle.mean(x, axis, keepdim)


def reduce_mean_wrapper(x, axis=0, keepdim=False, reduce_all=False):
    if reduce_all:
        return paddle.mean(x, list(range(len(x.shape))), keepdim)
    return paddle.mean(x, axis, keepdim)


class TestMeanOp(OpTest):
    def setUp(self):
        self.op_type = "mean"
        self.python_api = paddle.mean
        self.public_python_api = paddle.mean
        self.dtype = np.float64
        self.init_dtype_type()
        self.init_prim_type()
        self.inputs = {'X': np.random.random((10, 10)).astype(self.dtype)}
        self.outputs = {'Out': np.mean(self.inputs["X"])}

    def init_prim_type(self):
        self.prim_op_type = "comp"

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestMeanAllOp(OpTest):
    def setUp(self):
        self.op_type = "mean_all"
        self.python_api = paddle.mean_all
        self.public_python_api = paddle.mean_all
        self.dtype = np.float64
        self.init_dtype_type()
        self.init_prim_type()
        self.inputs = {'X': np.random.random((10, 10)).astype(self.dtype)}
        self.outputs = {'Out': np.mean(self.inputs["X"])}

    def init_prim_type(self):
        self.prim_op_type = "comp"

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestMeanOpPrim(TestMeanOp):
    def init_prim_type(self):
        self.prim_op_type = "prim"


class TestMeanOp_ZeroDim(OpTest):
    def setUp(self):
        self.op_type = "mean"
        self.python_api = paddle.mean
        self.dtype = np.float64
        self.public_python_api = paddle.mean
        self.init_prim_type()
        self.inputs = {'X': np.random.random([]).astype(self.dtype)}
        self.outputs = {'Out': np.mean(self.inputs["X"])}

    def init_prim_type(self):
        self.prim_op_type = "comp"

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestMeanOp_float64ZeroSize(OpTest):
    def setUp(self):
        self.op_type = "mean"
        self.python_api = paddle.mean
        self.dtype = np.float64
        self.public_python_api = paddle.mean
        self.inputs = {'X': np.array([]).astype(self.dtype)}
        self.outputs = {'Out': np.nan}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestMeanOp_float64ZeroSize3D(TestMeanOp_float64ZeroSize):
    def setUp(self):
        self.op_type = 'mean'
        self.python_api = paddle.mean
        self.dtype = np.float64
        self.public_python_api = paddle.mean
        self.shape = [2, 0, 4]

        x_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out_np = np.nan
        self.inputs = {'X': x_np}
        self.outputs = {'Out': out_np}


class TestMeanOp_Int32ZeroSize(OpTest):
    def setUp(self):
        self.op_type = "mean"
        self.python_api = paddle.mean
        self.dtype = np.int32
        self.public_python_api = paddle.mean
        self.inputs = {'X': np.array([]).astype(self.dtype)}
        self.outputs = {'Out': np.nan}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestMeanOp_Int64ZeroSize(OpTest):
    def setUp(self):
        self.op_type = "mean"
        self.python_api = paddle.mean
        self.dtype = np.int64
        self.public_python_api = paddle.mean
        self.inputs = {'X': np.array([]).astype(self.dtype)}
        self.outputs = {'Out': np.nan}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestMeanOp_Int64ZeroSize3D(TestMeanOp_Int64ZeroSize):
    def setUp(self):
        self.op_type = 'mean'
        self.python_api = paddle.mean
        self.dtype = np.int64
        self.public_python_api = paddle.mean
        self.shape = [2, 0, 4]

        x_np = np.random.uniform(0, 8, self.shape).astype(self.dtype)
        out_np = np.nan
        self.inputs = {'X': x_np}
        self.outputs = {'Out': out_np}


class TestMeanOp_Complex64ZeroSize(OpTest):
    def setUp(self):
        self.op_type = "mean"
        self.python_api = paddle.mean
        self.inputs = {'X': np.array([]).astype("complex64")}
        self.outputs = {'Out': np.mean(self.inputs["X"])}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestMeanOp_RealValuedNanInput(OpTest):
    def setUp(self):
        self.op_type = "mean"
        self.python_api = paddle.mean
        self.inputs = {'X': np.array([1, 2, 3, np.nan]).astype("float64")}
        self.outputs = {'Out': np.mean(self.inputs["X"])}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestMeanOp_RealNanInput(OpTest):
    def setUp(self):
        self.op_type = "mean"
        self.python_api = paddle.mean
        self.inputs = {
            'X': np.array([1 + 2j, 2 + 1j, np.nan + 1j]).astype("complex64")
        }
        self.outputs = {'Out': np.mean(self.inputs["X"])}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestMeanOp_ImagNanInput(OpTest):
    def setUp(self):
        self.op_type = "mean"
        self.python_api = paddle.mean
        self.inputs = {
            'X': np.array([1 + 1j, 2 + 2j, 1 + np.nan * 1j]).astype("complex64")
        }
        self.outputs = {'Out': np.mean(self.inputs["X"])}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestMeanAllOp_ZeroDim(OpTest):
    def setUp(self):
        self.op_type = "mean_all"
        self.python_api = paddle.mean_all
        self.dtype = np.float64
        self.public_python_api = paddle.mean_all
        self.init_prim_type()
        self.inputs = {'X': np.random.random([]).astype(self.dtype)}
        self.outputs = {'Out': np.mean(self.inputs["X"])}

    def init_prim_type(self):
        self.prim_op_type = "comp"

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestMeanOp_ZeroDim_Prim(TestMeanOp_ZeroDim):
    def init_prim_type(self):
        self.prim_op_type = "prim"


class TestMeanOpError(unittest.TestCase):
    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.int32)
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # The input type of mean_op must be Variable.
            input1 = 12
            self.assertRaises(TypeError, paddle.mean, input1)

            if paddle.is_compiled_with_cuda():
                input3 = paddle.static.data(
                    name='input3', shape=[-1, 4], dtype="float16"
                )
                paddle.nn.functional.softmax(input3)

        paddle.disable_static()


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFP16MeanOp(TestMeanOp):
    def init_dtype_type(self):
        self.dtype = np.float16
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_output_with_place(place, check_pir=True)

    def test_checkout_grad(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            with base.dygraph.guard():
                x_np = np.random.random((10, 10)).astype(self.dtype)
                x = paddle.to_tensor(x_np)
                x.stop_gradient = False
                y = paddle.mean(x)
                dx = paddle.grad(y, x)[0].numpy()
                dx_expected = self.dtype(1.0 / np.prod(x_np.shape)) * np.ones(
                    x_np.shape
                ).astype(self.dtype)
                np.testing.assert_array_equal(dx, dx_expected)


@OpTestTool.skip_if_not_cpu_bf16()
class TestBF16MeanOp(TestMeanOp):
    def init_dtype_type(self):
        self.dtype = np.uint16

    def test_check_output(self):
        paddle.enable_static()
        self.check_output_with_place(core.CPUPlace(), check_pir=True)

    def test_checkout_grad(self):
        place = core.CPUPlace()
        self.check_grad_with_place(place, ['X'], 'Out', check_pir=True)


def ref_reduce_mean(x, axis=None, keepdim=False, reduce_all=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    if reduce_all:
        axis = None
    return np.mean(x, axis=axis, keepdims=keepdim)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA",
)
class TestReduceMeanOp(OpTest):
    def setUp(self):
        self.op_type = 'reduce_mean'
        self.python_api = reduce_mean_wrapper
        self.public_python_api = reduce_mean_wrapper
        self.init_prim_type()
        self.dtype = 'float64'
        self.init_shapes()
        self.axis = [0]
        if self.shape == []:
            self.axis = []
        self.keepdim = False
        self.set_attrs()
        self.if_enable_cinn()

        np.random.seed(10)
        x_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if not hasattr(self, "reduce_all") and not x_np.shape == ():
            self.reduce_all = (not self.axis) or len(self.axis) == len(x_np)
        if x_np.shape == ():
            self.reduce_all = True
        out_np = ref_reduce_mean(x_np, self.axis, self.keepdim, self.reduce_all)
        self.inputs = {'X': x_np}
        self.outputs = {'Out': out_np}
        self.attrs = {
            'dim': self.axis,
            'keep_dim': self.keepdim,
            'reduce_all': self.reduce_all,
        }

    def init_prim_type(self):
        self.prim_op_type = "comp"

    def init_shapes(self):
        self.shape = [2, 3, 4, 5]

    def set_attrs(self):
        pass

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        if self.dtype != 'float16':
            self.check_output(
                check_prim=True, check_prim_pir=True, check_pir=True
            )
        else:
            place = paddle.CUDAPlace(0)
            self.check_output_with_place(
                place=place,
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
            )

    def test_check_grad(self):
        if self.dtype != 'float16':
            self.check_grad(
                ['X'],
                ['Out'],
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
            )
        else:
            place = paddle.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['X'],
                ['Out'],
                numeric_grad_delta=0.5,
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
            )


class TestReduceMeanOpPrim(TestReduceMeanOp):
    def init_prim_type(self):
        self.prim_op_type = "prim"

    def test_check_output(self):
        if self.dtype != 'float16':
            self.check_output(check_prim_pir=True, check_pir=True)
        else:
            place = paddle.CUDAPlace(0)
            self.check_output_with_place(
                place=place,
                check_prim_pir=True,
                check_pir=True,
            )

    def test_check_grad(self):
        if self.dtype != 'float16':
            self.check_grad(
                ['X'],
                ['Out'],
                check_prim_pir=True,
                check_pir=True,
            )
        else:
            place = paddle.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['X'],
                ['Out'],
                numeric_grad_delta=0.5,
                check_prim_pir=True,
                check_pir=True,
            )


class TestReduceMeanOp_ZeroDim(TestReduceMeanOp):
    def init_shapes(self):
        self.shape = []
        self.enable_cinn = False


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestReduceMeanBF16Op(OpTest):
    def setUp(self):
        self.op_type = 'reduce_mean'
        self.python_api = reduce_mean_wrapper
        self.public_python_api = reduce_mean_wrapper
        self.prim_op_type = "comp"
        self.dtype = np.uint16
        self.shape = [2, 3, 4, 5]
        self.axis = [0]
        self.keepdim = False
        self.set_attrs()
        self.if_enable_cinn()

        np.random.seed(10)
        x_np = np.random.uniform(-1, 1, self.shape).astype(np.float32)
        if not hasattr(self, "reduce_all"):
            self.reduce_all = (not self.axis) or len(self.axis) == len(x_np)

        out_np = ref_reduce_mean(x_np, self.axis, self.keepdim, self.reduce_all)
        self.inputs = {'X': convert_float_to_uint16(x_np)}
        self.outputs = {'Out': convert_float_to_uint16(out_np)}
        self.attrs = {
            'dim': self.axis,
            'keep_dim': self.keepdim,
            'reduce_all': self.reduce_all,
        }

    def if_enable_cinn(self):
        self.enable_cinn = False

    def set_attrs(self):
        pass

    def test_check_output(self):
        place = paddle.CUDAPlace(0)
        self.check_output_with_place(place, check_prim=True)

    def test_check_grad(self):
        place = paddle.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            ['Out'],
            numeric_grad_delta=0.05,
            check_prim=True,
            check_prim_pir=True,
        )


class TestReduceMeanOpDefaultAttrs(TestReduceMeanOp):
    def setUp(self):
        self.op_type = 'reduce_mean'
        self.python_api = reduce_mean_wrapper
        self.public_python_api = reduce_mean_wrapper
        self.prim_op_type = "comp"
        self.dtype = 'float64'
        self.shape = [2, 3, 4, 5]

        x_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out_np = np.mean(x_np, axis=0)
        self.inputs = {'X': x_np}
        self.outputs = {'Out': out_np}


class TestReduceMeanOpDefaultAttrsForPrim(TestReduceMeanOpPrim):
    def setUp(self):
        self.op_type = 'reduce_mean'
        self.python_api = reduce_mean_wrapper
        self.public_python_api = reduce_mean_wrapper
        self.init_prim_type()
        self.dtype = 'float64'
        self.shape = [2, 3, 4, 5]

        x_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out_np = np.mean(x_np, axis=0)
        self.inputs = {'X': x_np}
        self.outputs = {'Out': out_np}


class TestReduceMeanOpFloat32(TestReduceMeanOp):
    def set_attrs(self):
        self.dtype = 'float32'


class TestReduceMeanOpFloat32Prim(TestReduceMeanOpPrim):
    def set_attrs(self):
        self.dtype = 'float32'


class TestReduceMeanOpFloat16(TestReduceMeanOp):
    def set_attrs(self):
        self.dtype = 'float16'


class TestReduceMeanOpFloat16Prim(TestReduceMeanOpPrim):
    def set_attrs(self):
        self.dtype = 'float16'


class TestReduceMeanOpShape1D(TestReduceMeanOp):
    def set_attrs(self):
        self.shape = [100]


class TestReduceMeanOpShape1DFP16(TestReduceMeanOp):
    def set_attrs(self):
        self.shape = [100]
        self.dtype = 'float16'


class TestReduceMeanOpShape6D(TestReduceMeanOp):
    def set_attrs(self):
        self.shape = [2, 3, 4, 5, 6, 7]


class TestReduceMeanOpShape6DBF16(TestReduceMeanBF16Op):
    def set_attrs(self):
        self.shape = [2, 3, 4, 5, 6, 7]


class TestReduceMeanOpShape6DFP16(TestReduceMeanOp):
    def set_attrs(self):
        self.shape = [2, 3, 4, 5, 6, 7]
        self.dtype = 'float16'


class TestReduceMeanOpAxisAll(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]


class TestReduceMeanOpAxisAllPrim(TestReduceMeanOpPrim):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]


class TestReduceMeanOpAxisAllFP16(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]
        self.dtype = 'float16'


class TestReduceMeanOpAxisAllFP16Prim(TestReduceMeanOpPrim):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]
        self.dtype = 'float16'


class TestReduceMeanOpAxisAllBF16(TestReduceMeanBF16Op):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]


class TestReduceMeanOpAxisTuple(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = (0, 1, 2)


class TestReduceMeanOpAxisTupleFP16(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = (0, 1, 2)
        self.dtype = 'float16'


class TestReduceMeanOpAxisTupleBF16(TestReduceMeanBF16Op):
    def set_attrs(self):
        self.axis = (0, 1, 2)


class TestReduceMeanOpAxisNegative(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = [-2, -1]


class TestReduceMeanOpAxisNegativeFP16(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = [-2, -1]
        self.dtype = 'float16'


class TestReduceMeanOpAxisNegativeFP16Prim(TestReduceMeanOpPrim):
    def set_attrs(self):
        self.axis = [-2, -1]
        self.dtype = 'float16'


class TestReduceMeanOpAxisNegativeBF16(TestReduceMeanBF16Op):
    def set_attrs(self):
        self.axis = [-2, -1]


class TestReduceMeanOpKeepdimTrue1(TestReduceMeanOp):
    def set_attrs(self):
        self.keepdim = True


class TestReduceMeanOpKeepdimTrue1FP16(TestReduceMeanOp):
    def set_attrs(self):
        self.keepdim = True
        self.dtype = 'float16'


class TestReduceMeanOpKeepdimTrue1BF16(TestReduceMeanBF16Op):
    def set_attrs(self):
        self.keepdim = True


class TestReduceMeanOpKeepdimTrue2(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]
        self.keepdim = True


class TestReduceMeanOpKeepdimTrue2FP16(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]
        self.keepdim = True
        self.dtype = 'float16'


class TestReduceMeanOpKeepdimTrue2BF16(TestReduceMeanBF16Op):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]
        self.keepdim = True


class TestReduceMeanOpReduceAllTrue(TestReduceMeanOp):
    def set_attrs(self):
        self.reduce_all = True


class TestReduceMeanOpReduceAllTrueFP16(TestReduceMeanOp):
    def set_attrs(self):
        self.reduce_all = True
        self.dtype = 'float16'


class TestReduceMeanOpReduceAllTrueBF16(TestReduceMeanBF16Op):
    def set_attrs(self):
        self.reduce_all = True


class TestMeanAPI(unittest.TestCase):
    # test paddle.tensor.stat.mean

    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.x_shape)
            out1 = paddle.mean(x)
            out2 = paddle.tensor.mean(x)
            out3 = paddle.tensor.stat.mean(x)
            axis = np.arange(len(self.x_shape)).tolist()
            out4 = paddle.mean(x, axis)
            out5 = paddle.mean(x, tuple(axis))

            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={'X': self.x}, fetch_list=[out1, out2, out3, out4, out5]
            )
        out_ref = np.mean(self.x)
        for out in res:
            np.testing.assert_allclose(out, out_ref, rtol=0.0001)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)

        def test_case(x, axis=None, keepdim=False):
            x_tensor = paddle.to_tensor(x)
            out = paddle.mean(x_tensor, axis, keepdim)
            if isinstance(axis, list):
                axis = tuple(axis)
                if len(axis) == 0:
                    axis = None
            out_ref = np.mean(x, axis, keepdims=keepdim)
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.0001)

        test_case(self.x)
        test_case(self.x, [])
        test_case(self.x, -1)
        test_case(self.x, keepdim=True)
        test_case(self.x, 2, keepdim=True)
        test_case(self.x, [0, 2])
        test_case(self.x, (0, 2))
        test_case(self.x, [0, 1, 2, 3])
        paddle.enable_static()

    def test_base_api(self):
        with base.program_guard(base.Program(), base.Program()):
            x = paddle.static.data("x", shape=[10, 10], dtype="float32")
            out = paddle.mean(x=x, axis=1)
            place = base.CPUPlace()
            exe = base.Executor(place)
            x_np = np.random.rand(10, 10).astype(np.float32)
            res = exe.run(feed={"x": x_np}, fetch_list=[out])
        np.testing.assert_allclose(res[0], np.mean(x_np, axis=1), rtol=1e-05)

        with base.dygraph.guard():
            x_np = np.random.rand(10, 10).astype(np.float32)
            x = paddle.to_tensor(x_np)
            out = paddle.mean(x=x, axis=1)
        np.testing.assert_allclose(
            out.numpy(), np.mean(x_np, axis=1), rtol=1e-05
        )

    def test_errors(self):
        paddle.disable_static()
        x = np.random.uniform(-1, 1, [10, 12]).astype('float32')
        x = paddle.to_tensor(x)
        self.assertRaises(Exception, paddle.mean, x, -3)
        self.assertRaises(Exception, paddle.mean, x, 2)

        with self.assertRaises(Exception) as context:
            paddle.mean(x, axis=[0, 0])
        self.assertTrue(
            "Axis contains duplicate dimensions" in str(context.exception)
        )
        with self.assertRaises(Exception) as context:
            paddle.mean(x, axis=(1, 1))
        self.assertTrue(
            "Axis contains duplicate dimensions" in str(context.exception)
        )
        with self.assertRaises(Exception) as context:
            paddle.mean(x, axis=[-2, -2])
        self.assertTrue(
            "Axis contains duplicate dimensions" in str(context.exception)
        )
        with self.assertRaises(Exception) as context:
            paddle.mean(x, axis=[0, -2])
        self.assertTrue(
            "Axis contains duplicate dimensions" in str(context.exception)
        )


class TestMeanAPIInt32(unittest.TestCase):
    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.dtype = "int32"
        self.x_np = np.random.randint(-1, 10000, self.x_shape).astype(
            self.dtype
        )
        self.places = get_places()

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                x = paddle.to_tensor(self.x_np)
                out = paddle.mean(x=x)
            np.testing.assert_equal(
                out.numpy(),
                np.mean(self.x_np.astype("float32")).astype(self.dtype),
            )

    def test_static(self):
        paddle.enable_static()
        for place in self.places:
            with base.program_guard(base.Program(), base.Program()):
                x = paddle.static.data(
                    "x", shape=self.x_shape, dtype=self.dtype
                )
                out = paddle.mean(x=x)
                exe = base.Executor(place)
                res = exe.run(feed={"x": self.x_np}, fetch_list=[out])
            np.testing.assert_equal(
                res[0], np.mean(self.x_np.astype("float32")).astype(self.dtype)
            )


class TestMeanAPIInt64(TestMeanAPIInt32):
    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.dtype = "int64"
        self.x_np = np.random.randint(-1, 10000, self.x_shape).astype(
            self.dtype
        )
        self.places = get_places()


class TestMeanAPIBool(TestMeanAPIInt32):
    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.dtype = "bool"
        self.x_np = np.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        self.places = get_places()


class TestMeanWithTensorAxis1(TestReduceOPTensorAxisBase):
    def init_data(self):
        self.pd_api = paddle.mean
        self.np_api = np.mean
        self.x = paddle.randn([10, 5, 9, 9], dtype='float64')
        self.np_axis = np.array([1, 2], dtype='int64')
        self.tensor_axis = paddle.to_tensor([1, 2], dtype='int64')


class TestMeanWithTensorAxis2(TestReduceOPTensorAxisBase):
    def init_data(self):
        self.pd_api = paddle.mean
        self.np_api = np.mean
        self.x = paddle.randn([10, 10, 9, 9], dtype='float64')
        self.np_axis = np.array([0, 1, 2], dtype='int64')
        self.tensor_axis = [
            0,
            paddle.to_tensor([1], 'int64'),
            paddle.to_tensor([2], 'int64'),
        ]


class TestMeanDoubleGradCheck(unittest.TestCase):
    def mean_wrapper(self, x):
        return paddle.mean(x[0])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not include -1.
        eps = 0.005
        dtype = np.float32

        data = paddle.static.data('data', [3, 4, 5], dtype)
        data.persistable = True
        out = paddle.mean(data)
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.double_grad_check(
            [data], out, x_init=[data_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.mean_wrapper, [data], out, x_init=[data_arr], place=place
        )

    def test_grad(self):
        paddle.enable_static()
        for p in get_places():
            self.func(p)


class TestMeanTripleGradCheck(unittest.TestCase):
    def mean_wrapper(self, x):
        return paddle.mean(x[0])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not include -1.
        eps = 0.005
        dtype = np.float32

        data = paddle.static.data('data', [3, 4, 5], dtype)
        data.persistable = True
        out = paddle.mean(data)
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.triple_grad_check(
            [data], out, x_init=[data_arr], place=place, eps=eps
        )
        gradient_checker.triple_grad_check_for_dygraph(
            self.mean_wrapper, [data], out, x_init=[data_arr], place=place
        )

    def test_grad(self):
        paddle.enable_static()
        for p in get_places():
            self.func(p)


class TestMeanOp_ZeroSize(OpTest):
    def setUp(self):
        self.op_type = "mean"
        self.python_api = paddle.mean
        self.dtype = np.float64
        self.public_python_api = paddle.mean
        self.init_prim_type()
        self.inputs = {'X': np.random.random([2, 0, 2, 2]).astype(self.dtype)}
        self.outputs = {'Out': np.mean(self.inputs["X"])}

    def init_prim_type(self):
        self.prim_op_type = "comp"

    def test_check_output(self):
        self.check_output(check_pir=True, equal_nan=True)

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestMeanOp_ZeroSize2(OpTest):
    def setUp(self):
        self.op_type = 'reduce_mean'
        self.python_api = reduce_mean_wrapper
        self.public_python_api = reduce_mean_wrapper
        self.init_prim_type()
        self.dtype = 'float64'
        self.init_shapes()
        self.axis = [0]
        if self.shape == []:
            self.axis = []
        self.keepdim = False
        self.set_attrs()
        self.if_enable_cinn()

        np.random.seed(10)
        x_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if not hasattr(self, "reduce_all") and not x_np.shape == ():
            self.reduce_all = (not self.axis) or len(self.axis) == len(x_np)
        if x_np.shape == ():
            self.reduce_all = True
        out_np = ref_reduce_mean(x_np, self.axis, self.keepdim, self.reduce_all)
        self.inputs = {'X': x_np}
        self.outputs = {'Out': out_np}
        self.attrs = {
            'dim': self.axis,
            'keep_dim': self.keepdim,
            'reduce_all': self.reduce_all,
        }

    def init_prim_type(self):
        self.prim_op_type = "comp"

    def init_shapes(self):
        self.shape = [2, 0, 2, 2]

    def set_attrs(self):
        pass

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        if self.dtype != 'float16':
            self.check_output(
                check_prim=True, check_prim_pir=True, check_pir=True
            )
        else:
            place = paddle.CUDAPlace(0)
            self.check_output_with_place(
                place=place,
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
            )

    def test_check_grad(self):
        if self.dtype != 'float16':
            self.check_grad(
                ['X'],
                ['Out'],
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
            )
        else:
            place = paddle.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['X'],
                ['Out'],
                numeric_grad_delta=0.5,
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
            )


class TestMeanOp_ZeroSize3(OpTest):
    def setUp(self):
        self.op_type = 'mean'
        self.python_api = paddle.mean
        self.init_prim_type()
        self.dtype = 'float64'
        self.shape = [2, 0, 4]
        self.axis = 1
        self.keepdim = False
        self.set_attrs()

        self.inputs = {'X': np.array([], dtype=self.dtype).reshape(self.shape)}
        self.outputs = {
            'Out': np.mean(
                self.inputs["X"], axis=self.axis, keepdims=self.keepdim
            )
        }

    def set_attrs(self):
        pass

    def init_prim_type(self):
        self.prim_op_type = "prim"

    def test_check_output(self):
        self.check_output(check_pir=True, equal_nan=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


def skip_if_xpu_or_onednn_and_not_float32(dtype):
    """Skip test if using XPU or OneDNN and dtype is not float32"""

    def decorator(test_func):
        def wrapper(self):
            # Check if we're using XPU
            is_xpu = (hasattr(self, 'use_xpu') and self.use_xpu) or (
                paddle.device.get_device().startswith('xpu')
            )

            # Check if we're using OneDNN
            is_onednn = base.core.globals().get("FLAGS_use_onednn", False) or (
                hasattr(self, 'use_onednn') and self.use_onednn
            )

            # Skip if using XPU or OneDNN and dtype is not float32
            if (is_xpu or is_onednn) and dtype != 'float32':
                self.skipTest(
                    f"Skip {dtype} test for XPU/OneDNN, only test float32"
                )

            return test_func(self)

        return wrapper

    return decorator


class TestMeanDtypeParameter(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_data = np.random.rand(3, 4, 5).astype('float32')

    def tearDown(self):
        paddle.enable_static()

    def test_dtype_float32(self):
        x = paddle.to_tensor(self.x_data)
        result = paddle.mean(x, dtype='float32')
        self.assertEqual(result.dtype, paddle.float32)

    def test_dtype_float32_backward(self):
        x = paddle.to_tensor(self.x_data, stop_gradient=False)
        result = paddle.mean(x, dtype='float32')
        result.backward()

        # Check gradient shape matches input shape
        self.assertEqual(x.grad.shape, x.shape)
        # Check gradient values (should be 1/numel for mean)
        expected_grad = np.ones_like(self.x_data) / self.x_data.size
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_dtype_float64(self):
        x = paddle.to_tensor(self.x_data)
        result = paddle.mean(x, dtype='float64')
        self.assertEqual(result.dtype, paddle.float64)

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_dtype_float64_backward(self):
        x = paddle.to_tensor(self.x_data, stop_gradient=False)
        result = paddle.mean(x, dtype='float64')
        result.backward()

        self.assertEqual(x.grad.shape, x.shape)
        expected_grad = np.ones_like(self.x_data) / self.x_data.size
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)

    def test_dtype_none_default(self):
        x = paddle.to_tensor(self.x_data)
        result1 = paddle.mean(x, dtype=None)
        result2 = paddle.mean(x)
        self.assertEqual(result1.dtype, result2.dtype)
        np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-05)

    def test_dtype_none_default_backward(self):
        x1 = paddle.to_tensor(self.x_data, stop_gradient=False)
        x2 = paddle.to_tensor(self.x_data, stop_gradient=False)

        result1 = paddle.mean(x1, dtype=None)
        result2 = paddle.mean(x2)

        result1.backward()
        result2.backward()

        # Gradients should be identical
        np.testing.assert_allclose(x1.grad.numpy(), x2.grad.numpy(), rtol=1e-05)

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_dtype_with_axis(self):
        x = paddle.to_tensor(self.x_data)
        result = paddle.mean(x, axis=1, dtype='float64')
        self.assertEqual(result.dtype, paddle.float64)
        self.assertEqual(result.shape, [3, 5])

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_dtype_with_axis_backward(self):
        x = paddle.to_tensor(self.x_data, stop_gradient=False)
        result = paddle.mean(x, axis=1, dtype='float64')
        loss = paddle.sum(result)
        loss.backward()

        # Check gradient shape
        self.assertEqual(x.grad.shape, x.shape)
        # For mean along axis=1, gradient should be 1/axis_size for each element
        expected_grad = np.ones_like(self.x_data) / self.x_data.shape[1]
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)


class TestMeanOutParameter(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_data = np.random.rand(3, 4, 5).astype('float32')

    def tearDown(self):
        paddle.enable_static()

    def test_out_parameter_basic(self):
        x = paddle.to_tensor(self.x_data)
        out = paddle.empty([], dtype='float32')
        result = paddle.mean(x, out=out)

        # Check that out is modified in-place
        self.assertTrue(paddle.allclose(out, result))
        np.testing.assert_allclose(
            out.numpy(), np.mean(self.x_data), rtol=1e-05
        )

    def test_out_parameter_basic_backward(self):
        x = paddle.to_tensor(self.x_data, stop_gradient=False)
        out = paddle.empty([], dtype='float32')
        result = paddle.mean(x, out=out)
        result.backward()

        self.assertEqual(x.grad.shape, x.shape)
        expected_grad = np.ones_like(self.x_data) / self.x_data.size
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)

    def test_out_parameter_with_axis(self):
        x = paddle.to_tensor(self.x_data)
        out = paddle.empty([3, 5], dtype='float32')
        result = paddle.mean(x, axis=1, out=out)

        self.assertTrue(paddle.allclose(out, result))
        self.assertEqual(out.shape, [3, 5])

    def test_out_parameter_with_axis_backward(self):
        x = paddle.to_tensor(self.x_data, stop_gradient=False)
        out = paddle.empty([3, 5], dtype='float32')
        result = paddle.mean(x, axis=1, out=out)
        loss = paddle.sum(result)
        loss.backward()

        self.assertEqual(x.grad.shape, x.shape)
        expected_grad = np.ones_like(self.x_data) / self.x_data.shape[1]
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)

    def test_out_parameter_with_keepdim(self):
        x = paddle.to_tensor(self.x_data)
        out = paddle.empty([1, 1, 1], dtype='float32')
        result = paddle.mean(x, axis=[0, 1, 2], keepdim=True, out=out)

        self.assertTrue(paddle.allclose(out, result))
        self.assertEqual(out.shape, [1, 1, 1])

    def test_out_parameter_with_keepdim_backward(self):
        x = paddle.to_tensor(self.x_data, stop_gradient=False)
        out = paddle.empty([1, 1, 1], dtype='float32')
        result = paddle.mean(x, axis=[0, 1, 2], keepdim=True, out=out)
        result.backward()

        self.assertEqual(x.grad.shape, x.shape)
        expected_grad = np.ones_like(self.x_data) / self.x_data.size
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)

    def test_out_parameter_none_default(self):
        x = paddle.to_tensor(self.x_data)
        result1 = paddle.mean(x, out=None)
        result2 = paddle.mean(x)

        self.assertEqual(result1.dtype, result2.dtype)
        np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-05)

    def test_out_parameter_none_default_backward(self):
        x1 = paddle.to_tensor(self.x_data, stop_gradient=False)
        x2 = paddle.to_tensor(self.x_data, stop_gradient=False)

        result1 = paddle.mean(x1, out=None)
        result2 = paddle.mean(x2)

        result1.backward()
        result2.backward()

        np.testing.assert_allclose(x1.grad.numpy(), x2.grad.numpy(), rtol=1e-05)


class TestMeanDtypeAndOutCombination(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_data = np.random.rand(2, 3, 4).astype('float32')

    def tearDown(self):
        paddle.enable_static()

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_dtype_and_out_compatible(self):
        x = paddle.to_tensor(self.x_data)
        out = paddle.empty([], dtype='float64')
        result = paddle.mean(x, dtype='float64', out=out)

        self.assertEqual(out.dtype, paddle.float64)
        self.assertEqual(result.dtype, paddle.float64)
        self.assertTrue(paddle.allclose(out, result))

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_dtype_and_out_compatible_backward(self):
        x = paddle.to_tensor(self.x_data, stop_gradient=False)
        out = paddle.empty([], dtype='float64')
        result = paddle.mean(x, dtype='float64', out=out)
        result.backward()

        self.assertEqual(x.grad.shape, x.shape)
        expected_grad = np.ones_like(self.x_data) / self.x_data.size
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)

    def test_dtype_and_out_with_keepdim(self):
        x = paddle.to_tensor(self.x_data)
        out = paddle.empty([2, 1, 4], dtype='float32')
        result = paddle.mean(x, axis=1, keepdim=True, dtype='float32', out=out)

        self.assertEqual(out.shape, [2, 1, 4])
        self.assertTrue(paddle.allclose(out, result))

    def test_dtype_and_out_with_keepdim_backward(self):
        x = paddle.to_tensor(self.x_data, stop_gradient=False)
        out = paddle.empty([2, 1, 4], dtype='float32')
        result = paddle.mean(x, axis=1, keepdim=True, dtype='float32', out=out)
        loss = paddle.sum(result)
        loss.backward()

        self.assertEqual(x.grad.shape, x.shape)
        expected_grad = np.ones_like(self.x_data) / self.x_data.shape[1]
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)


class TestMeanParameterAlias(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_data = np.random.rand(3, 4, 5).astype('float32')

    def tearDown(self):
        paddle.enable_static()

    def test_x_alias_input(self):
        # Test x parameter alias
        x = paddle.to_tensor(self.x_data)
        result1 = paddle.mean(x=x, axis=1)
        result2 = paddle.mean(input=x, axis=1)

        np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-05)

    def test_x_alias_input_backward(self):
        x1 = paddle.to_tensor(self.x_data, stop_gradient=False)
        x2 = paddle.to_tensor(self.x_data, stop_gradient=False)

        result1 = paddle.mean(x=x1, axis=1)
        result2 = paddle.mean(input=x2, axis=1)

        loss1 = paddle.sum(result1)
        loss2 = paddle.sum(result2)

        loss1.backward()
        loss2.backward()

        np.testing.assert_allclose(x1.grad.numpy(), x2.grad.numpy(), rtol=1e-05)

    def test_axis_alias_dim(self):
        # Test axis parameter alias
        x = paddle.to_tensor(self.x_data)
        result1 = paddle.mean(x, axis=1)
        result2 = paddle.mean(x, dim=1)

        np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-05)

    def test_axis_alias_dim_backward(self):
        x1 = paddle.to_tensor(self.x_data, stop_gradient=False)
        x2 = paddle.to_tensor(self.x_data, stop_gradient=False)

        result1 = paddle.mean(x1, axis=1)
        result2 = paddle.mean(x2, dim=1)

        loss1 = paddle.sum(result1)
        loss2 = paddle.sum(result2)

        loss1.backward()
        loss2.backward()

        np.testing.assert_allclose(x1.grad.numpy(), x2.grad.numpy(), rtol=1e-05)

    def test_multiple_axis_alias(self):
        x = paddle.to_tensor(self.x_data)
        result1 = paddle.mean(x, axis=[0, 2])
        result2 = paddle.mean(x, dim=[0, 2])

        np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-05)

    def test_multiple_axis_alias_backward(self):
        x1 = paddle.to_tensor(self.x_data, stop_gradient=False)
        x2 = paddle.to_tensor(self.x_data, stop_gradient=False)

        result1 = paddle.mean(x1, axis=[0, 2])
        result2 = paddle.mean(x2, dim=[0, 2])

        loss1 = paddle.sum(result1)
        loss2 = paddle.sum(result2)

        loss1.backward()
        loss2.backward()

        np.testing.assert_allclose(x1.grad.numpy(), x2.grad.numpy(), rtol=1e-05)

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_alias_with_dtype_and_out(self):
        x = paddle.to_tensor(self.x_data)
        out1 = paddle.empty([4], dtype='float64')
        out2 = paddle.empty([4], dtype='float64')

        result1 = paddle.mean(input=x, axis=[0, 2], dtype='float64', out=out1)
        result2 = paddle.mean(x=x, dim=[0, 2], dtype='float64', out=out2)

        np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-05)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), rtol=1e-05)

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_alias_with_dtype_and_out_backward(self):
        x1 = paddle.to_tensor(self.x_data, stop_gradient=False)
        x2 = paddle.to_tensor(self.x_data, stop_gradient=False)

        out1 = paddle.empty([4], dtype='float64')
        out2 = paddle.empty([4], dtype='float64')

        result1 = paddle.mean(input=x1, axis=[0, 2], dtype='float64', out=out1)
        result2 = paddle.mean(x=x2, dim=[0, 2], dtype='float64', out=out2)

        loss1 = paddle.sum(result1)
        loss2 = paddle.sum(result2)

        loss1.backward()
        loss2.backward()

        np.testing.assert_allclose(x1.grad.numpy(), x2.grad.numpy(), rtol=1e-05)


class TestMeanNewParametersStatic(unittest.TestCase):
    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_static_dtype_parameter(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()

        with paddle.static.program_guard(main_prog, startup_prog):
            x = paddle.static.data('x', shape=[3, 4], dtype='float32')
            result = paddle.mean(x, dtype='float64')

            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
            exe = base.Executor(place)

            exe.run(startup_prog)
            x_np = np.random.rand(3, 4).astype('float32')
            out = exe.run(main_prog, feed={'x': x_np}, fetch_list=[result])

            expected = np.mean(x_np).astype('float64')
            np.testing.assert_allclose(out[0], expected, rtol=1e-05)

    def test_static_alias_parameters(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()

        with paddle.static.program_guard(main_prog, startup_prog):
            x = paddle.static.data('x', shape=[3, 4], dtype='float32')
            result1 = paddle.mean(input=x, dim=1)
            result2 = paddle.mean(x=x, axis=1)

            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
            exe = base.Executor(place)

            exe.run(startup_prog)
            x_np = np.random.rand(3, 4).astype('float32')
            out1, out2 = exe.run(
                main_prog, feed={'x': x_np}, fetch_list=[result1, result2]
            )

            np.testing.assert_allclose(out1, out2, rtol=1e-05)


class TestMeanBoundaryConditions(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def test_dtype_with_int_input(self):
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='int32')
        result = paddle.mean(x, dtype='float32')
        self.assertEqual(result.dtype, paddle.float32)
        expected = 3.5
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-05)

    def test_dtype_with_int_input_backward(self):
        # Int input tensors don't support gradients, so we test the conversion
        x_float = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype='float32',
            stop_gradient=False,
        )
        result = paddle.mean(x_float, dtype='float32')
        result.backward()

        self.assertEqual(x_float.grad.shape, x_float.shape)
        expected_grad = np.ones_like(x_float.numpy()) / x_float.numel()
        np.testing.assert_allclose(
            x_float.grad.numpy(), expected_grad, rtol=1e-05
        )

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_all_parameters_combination(self):
        # Test all new parameters together
        x_data = np.random.rand(2, 3, 4).astype('float32')
        x = paddle.to_tensor(x_data)
        out = paddle.empty([2, 4], dtype='float64')

        result = paddle.mean(
            input=x, dim=1, keepdim=False, dtype='float64', out=out
        )

        self.assertEqual(result.dtype, paddle.float64)
        self.assertEqual(result.shape, [2, 4])
        self.assertTrue(paddle.allclose(out, result))

        expected = np.mean(x_data, axis=1).astype('float64')
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-05)

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_all_parameters_combination_backward(self):
        x_data = np.random.rand(2, 3, 4).astype('float32')
        x = paddle.to_tensor(x_data, stop_gradient=False)
        out = paddle.empty([2, 4], dtype='float64')

        result = paddle.mean(
            input=x, dim=1, keepdim=False, dtype='float64', out=out
        )

        loss = paddle.sum(result)
        loss.backward()

        self.assertEqual(x.grad.shape, x.shape)
        expected_grad = np.ones_like(x_data) / x_data.shape[1]
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
