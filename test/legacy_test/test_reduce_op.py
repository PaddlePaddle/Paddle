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

import os
import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci
from utils import dygraph_guard, static_guard

import paddle
from paddle import base
from paddle.base import core
from paddle.base.framework import in_pir_mode


class TestSumOp(OpTest):
    def setUp(self):
        self.init_dtype()
        self.init_input()
        self.init_attrs()
        self.calc_output()

        self.python_api = paddle.sum
        self.public_python_api = paddle.sum
        self.op_type = "reduce_sum"
        self.prim_op_type = "prim"
        self.inputs = {'X': self.x}
        self.outputs = {'Out': self.out}
        self.if_enable_cinn()

    def init_dtype(self):
        self.dtype = np.float64

    def init_input(self):
        self.x = np.random.random((5, 6, 10)).astype(self.dtype)

    def init_attrs(self):
        self.attrs = {'dim': [0]}

    def if_enable_cinn(self):
        pass

    def calc_output(self):
        self.out = self.x.sum(axis=tuple(self.attrs['dim']))

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestComplexSumOP(TestSumOp):
    def init_dtype(self):
        self.dtype = np.complex128

    def init_input(self):
        self.x = np.random.random((3, 4)).astype(self.dtype)

    def init_attrs(self):
        self.attrs = {'dim': [0]}

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim=False)


class TestSumOp_ZeroDim(TestSumOp):
    def init_attrs(self):
        self.attrs = {'dim': []}

    def init_input(self):
        self.x = np.random.random([]).astype(self.dtype)

    def calc_output(self):
        self.out = self.x.sum(axis=None)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_pir=True,
            check_prim=True,
            check_prim_pir=True,
        )


class TestSumOp5D(TestSumOp):
    def init_input(self):
        self.x = np.random.random((1, 2, 5, 6, 10)).astype(self.dtype)

    def init_attrs(self):
        self.attrs = {'dim': [0]}


class TestSumOp6D(TestSumOp):
    def init_input(self):
        self.x = np.random.random((1, 1, 2, 5, 6, 10)).astype(self.dtype)

    def init_attrs(self):
        self.attrs = {'dim': [0]}


class TestSumOp8D(TestSumOp):
    def init_input(self):
        self.x = np.random.random((1, 3, 1, 2, 1, 4, 3, 10)).astype(self.dtype)

    def init_attrs(self):
        self.attrs = {'dim': (0, 3)}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)


class TestSumOp_withInt(TestSumOp):
    def init_input(self):
        # ref to https://en.wikipedia.org/wiki/Half-precision_floating-point_format
        # Precision limitations on integer values between 0 and 2048 can be exactly represented
        self.x = np.random.randint(0, 30, (10, 10)).astype(self.dtype)

    def init_attrs(self):
        self.attrs = {'dim': (0, 1)}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def calc_gradient(self):
        x = self.inputs["X"]
        grad = np.ones(x.shape, dtype=x.dtype)
        return (grad,)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            user_defined_grads=self.calc_gradient(),
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )


class TestSumOp3Dim(TestSumOp):
    def init_input(self):
        self.x = np.random.uniform(0, 0.1, (5, 6, 10)).astype(self.dtype)

    def init_attrs(self):
        self.attrs = {'dim': (0, 1, 2)}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def calc_gradient(self):
        x = self.inputs["X"]
        grad = np.ones(x.shape, dtype=x.dtype)
        return (grad,)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            user_defined_grads=self.calc_gradient(),
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )


def create_test_fp16_class(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestSumOpFp16(parent):
        def init_dtype(self):
            self.dtype = np.float16

        def test_check_output(self):
            self.check_output(check_pir=True)

        def test_check_grad(self):
            self.check_grad(
                ['X'],
                'Out',
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
            )


create_test_fp16_class(TestSumOp)
create_test_fp16_class(TestSumOp_ZeroDim)
create_test_fp16_class(TestSumOp5D)
create_test_fp16_class(TestSumOp6D)
create_test_fp16_class(TestSumOp8D)
create_test_fp16_class(TestSumOp_withInt)
create_test_fp16_class(TestSumOp3Dim)


def create_test_bf16_class(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda() or paddle.is_compiled_with_rocm(),
        "core is not compiled with CUDA",
    )
    class TestSumOpBf16(parent):
        def setUp(self):
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            self.outputs = {'Out': convert_float_to_uint16(self.out)}
            self.enable_cinn = False

        def init_dtype(self):
            self.dtype = np.uint16

        def test_check_output(self):
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, check_pir=True)

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['X'],
                'Out',
                user_defined_grads=self.gradient,
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
            )

        def calc_gradient(self):
            x = self.x
            grad = np.ones(x.shape, dtype=x.dtype)
            return [grad]


create_test_bf16_class(TestSumOp)
create_test_bf16_class(TestSumOp_ZeroDim)
create_test_bf16_class(TestSumOp5D)
create_test_bf16_class(TestSumOp6D)
create_test_bf16_class(TestSumOp8D)
create_test_bf16_class(TestSumOp_withInt)
create_test_bf16_class(TestSumOp3Dim)


class TestSumAPIZeroDimKeepDim(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_static(self):
        for place in self.places:
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with base.program_guard(main, startup):
                input = paddle.static.data(
                    name="input", shape=[0, 0], dtype="float32"
                )
                result = paddle.sum(x=input, keepdim=True)
                input_np = np.random.rand(0, 0).astype("float32")
                exe = paddle.static.Executor(place)
                fetches = exe.run(
                    main,
                    feed={"input": input_np},
                    fetch_list=[result],
                )
                self.assertEqual(fetches[0].shape, (1, 1))
                np.allclose(fetches[0], np.sum(input_np, keepdims=True))

    def test_dygraph(self):
        paddle.disable_static()
        for place in self.places:
            with base.dygraph.guard(place):
                np_x = np.random.rand(0, 0).astype("float32")
                x = paddle.to_tensor(np_x)

                out1 = paddle.sum(x, keepdim=True)
                np_out1 = out1.numpy()
                expect_res1 = np.sum(np_x, keepdims=True)
                np.allclose(np_out1, expect_res1)

                out2 = paddle.sum(x, axis=0, keepdim=True)
                np_out2 = out2.numpy()
                expect_res2 = np.sum(np_x, axis=0, keepdims=True)
                np.allclose(np_out2, expect_res2)

                out3 = paddle.sum(x, axis=-1, keepdim=True)
                np_out3 = out3.numpy()
                expect_res3 = np.sum(np_x, axis=-1, keepdims=True)
                np.allclose(np_out3, expect_res3)

        paddle.enable_static()


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework."
)
class TestMaxOp(OpTest):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.prim_op_type = "prim"
        self.python_api = paddle.max
        self.public_python_api = paddle.max
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [-1]}
        self.outputs = {
            'Out': self.inputs['X'].max(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        # only composite op support gradient check of reduce_max
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            only_check_prim=True,
            check_pir=True,
        )


class TestMaxOp_ZeroDim(OpTest):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.prim_op_type = "prim"
        self.python_api = paddle.max
        self.public_python_api = paddle.max
        self.if_enable_cinn()
        self.init_inputs_and_outputs()

    def if_enable_cinn(self):
        self.enable_cinn = False

    def init_inputs_and_outputs(self):
        self.inputs = {'X': np.random.random([]).astype("float64")}
        self.attrs = {'dim': []}
        self.outputs = {
            'Out': self.inputs['X'].max(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        # only composite op support gradient check of reduce_max
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            only_check_prim=True,
            check_pir=True,
        )


class TestMaxOp_ZeroDim1(TestMaxOp_ZeroDim):
    def init_inputs_and_outputs(self):
        self.inputs = {'X': np.random.random([5]).astype("float64")}
        self.attrs = {'dim': [0]}
        self.outputs = {'Out': self.inputs['X'].max(axis=(0,))}


class TestMaxOp_ZeroDim2(TestMaxOp_ZeroDim1):
    def init_inputs_and_outputs(self):
        self.inputs = {'X': np.random.random([5, 20]).astype("float64")}
        self.attrs = {'dim': [0, 1]}
        self.outputs = {'Out': self.inputs['X'].max(axis=(0, 1))}


class TestMaxFP32Op(OpTest):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.prim_op_type = "prim"
        self.python_api = paddle.max
        self.public_python_api = paddle.max
        self.init_dtype()
        self.if_enable_cinn()
        if self.dtype == np.uint16:
            x = np.random.random((5, 6, 10)).astype(np.float32)
            self.inputs = {'X': convert_float_to_uint16(x)}
        else:
            x = np.random.random((5, 6, 10)).astype(self.dtype)
            self.inputs = {'X': x}
        self.attrs = {'dim': [-1], 'keep_dim': True}
        out = x.max(axis=tuple(self.attrs['dim']), keepdims=True)
        if self.dtype == np.uint16:
            self.outputs = {'Out': convert_float_to_uint16(out)}
        else:
            self.outputs = {'Out': out}

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        # only composite op support gradient check of reduce_max
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            only_check_prim=True,
            check_pir=True,
        )

    def init_dtype(self):
        self.dtype = np.float32


class TestMaxFP16Op(TestMaxFP32Op):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or paddle.is_compiled_with_rocm()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestMaxBF16Op(TestMaxFP32Op):
    def init_dtype(self):
        self.dtype = np.uint16

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        self.check_output_with_place(core.CUDAPlace(0), check_pir=True)

    def test_check_grad(self):
        # only composite op support gradient check of reduce_max
        self.check_grad_with_place(
            core.CUDAPlace(0),
            ['X'],
            'Out',
            check_prim=True,
            only_check_prim=True,
            check_pir=True,
        )


@skip_check_grad_ci(
    reason="reduce_min is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework."
)
class TestMinOp(OpTest):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.python_api = paddle.min
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [2]}
        self.outputs = {
            'Out': self.inputs['X'].min(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestMinOp_ZeroDim(OpTest):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.python_api = paddle.min
        self.inputs = {'X': np.random.random([]).astype("float64")}
        self.attrs = {'dim': []}
        self.outputs = {
            'Out': self.inputs['X'].min(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestMin6DOp(OpTest):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.python_api = paddle.min
        self.inputs = {
            'X': np.random.random((2, 4, 3, 5, 6, 10)).astype("float64")
        }
        self.attrs = {'dim': [2, 4]}
        self.outputs = {
            'Out': self.inputs['X'].min(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestMin8DOp(OpTest):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.python_api = paddle.min
        self.inputs = {
            'X': np.random.random((2, 4, 3, 5, 6, 3, 2, 4)).astype("float64")
        }
        self.attrs = {'dim': [2, 3, 4]}
        self.outputs = {
            'Out': self.inputs['X'].min(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


@skip_check_grad_ci(
    reason="reduce_min is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework."
)
@unittest.skipIf(
    paddle.is_compiled_with_rocm(), "ROCm doesn't have FP16 reduce_min kernel"
)
class TestMinFP16Op(OpTest):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.python_api = paddle.min
        self.public_python_api = paddle.min
        self.init_dtype()
        if self.dtype == np.uint16:
            x = np.random.random((5, 6, 10)).astype(np.float32)
            self.inputs = {'X': convert_float_to_uint16(x)}
        else:
            x = np.random.random((5, 6, 10)).astype(self.dtype)
            self.inputs = {'X': x}
        self.attrs = {'dim': [2], 'keep_dim': True}
        out = x.min(axis=tuple(self.attrs['dim']), keepdims=True)
        if self.dtype == np.uint16:
            self.outputs = {'Out': convert_float_to_uint16(out)}
        else:
            self.outputs = {'Out': out}

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(check_pir=True)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or paddle.is_compiled_with_rocm()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestMinBF16Op(TestMinFP16Op):
    def init_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        self.check_output_with_place(core.CUDAPlace(0), check_pir=True)


def raw_reduce_prod(x, dim=[0], keep_dim=False):
    return paddle.prod(x, dim, keep_dim)


class TestProdOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.python_api = raw_reduce_prod
        self.public_python_api = raw_reduce_prod
        self.prim_op_type = "prim"
        self.init_data_type()
        self.init_inputs_and_outputs()
        self.if_enable_cinn()

    def init_inputs_and_outputs(self):
        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.data_type)}
        self.outputs = {'Out': self.inputs['X'].prod(axis=0)}

    def init_data_type(self):
        self.data_type = (
            "float32" if core.is_compiled_with_rocm() else "float64"
        )

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True
        )


@unittest.skipIf(
    not paddle.is_compiled_with_cuda(), "FP16 test runs only on GPU"
)
class TestProdFP16OP(TestProdOp):
    def init_data_type(self):
        self.data_type = "float16"

    def test_check_output(self):
        self.check_output_with_place(place=paddle.CUDAPlace(0), check_pir=True)

    def test_check_grad(self):
        self.check_grad_with_place(
            paddle.CUDAPlace(0),
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or paddle.is_compiled_with_rocm()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestProdBFP16OP(TestProdOp):
    def init_data_type(self):
        self.data_type = np.uint16

    def init_inputs_and_outputs(self):
        x = np.random.random((5, 6, 10)).astype("float32")
        out = x.prod(axis=0)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        self.check_output_with_place(place=paddle.CUDAPlace(0), check_pir=True)

    def test_check_grad(self):
        self.check_grad_with_place(
            paddle.CUDAPlace(0),
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestProdOpFp64(TestProdOp):
    def init_data_type(self):
        self.data_type = "float64"


class TestProdOp_ZeroDim(OpTest):
    def setUp(self):
        self.python_api = raw_reduce_prod
        self.public_python_api = raw_reduce_prod
        self.op_type = "reduce_prod"
        self.prim_op_type = "prim"
        self.init_inputs_and_outputs()
        # 0-D tensor doesn't support in cinn
        self.enable_cinn = False

    def init_inputs_and_outputs(self):
        self.inputs = {'X': np.random.random([]).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].prod()}
        self.attrs = {'dim': [], 'reduce_all': True}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True
        )


class TestProdOp_ZeroDim1(TestProdOp):
    def setUp(self):
        self.python_api = paddle.prod
        self.public_python_api = paddle.prod
        self.op_type = "reduce_prod"
        self.prim_op_type = "prim"
        self.init_inputs_and_outputs()
        # 0-D tensor doesn't support in cinn
        self.enable_cinn = False

    def init_inputs_and_outputs(self):
        self.inputs = {'X': np.random.random([100]).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].prod()}
        self.attrs = {'dim': [], 'reduce_all': True}


class TestProdOp_ZeroDim2(TestProdOp_ZeroDim1):
    def init_inputs_and_outputs(self):
        self.inputs = {'X': np.random.random([5, 6, 10]).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].prod()}
        self.attrs = {'dim': [], 'reduce_all': True}


class TestProd6DOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.python_api = raw_reduce_prod
        self.public_python_api = raw_reduce_prod
        self.prim_op_type = "prim"
        self.init_data_type()
        self.init_inputs_and_outputs()
        self.if_enable_cinn()

    def init_data_type(self):
        self.data_type = (
            "float32" if core.is_compiled_with_rocm() else "float64"
        )

    def init_inputs_and_outputs(self):
        self.inputs = {
            'X': np.random.random((5, 6, 2, 3, 4, 2)).astype(self.data_type)
        }
        self.attrs = {'dim': [2, 3, 4]}
        self.outputs = {
            'Out': self.inputs['X'].prod(axis=tuple(self.attrs['dim']))
        }

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim=True, check_pir=True)


@unittest.skipIf(
    not paddle.is_compiled_with_cuda(), "FP16 test runs only on GPU"
)
class TestProd6DFP16OP(TestProd6DOp):
    def init_data_type(self):
        self.data_type = "float16"

    def test_check_output(self):
        self.check_output_with_place(place=paddle.CUDAPlace(0), check_pir=True)

    def test_check_grad(self):
        self.check_grad_with_place(
            paddle.CUDAPlace(0), ['X'], 'Out', check_prim=True, check_pir=True
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or paddle.is_compiled_with_rocm()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestProd6DBFP16OP(TestProd6DOp):
    def init_data_type(self):
        self.data_type = np.uint16

    def init_inputs_and_outputs(self):
        x = np.random.random((5, 6, 2, 3, 4, 2)).astype("float32")
        self.attrs = {'dim': [2, 3, 4]}
        out = x.prod(axis=tuple(self.attrs['dim']))
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        self.check_output_with_place(place=paddle.CUDAPlace(0), check_pir=True)

    def test_check_grad(self):
        self.check_grad_with_place(
            paddle.CUDAPlace(0), ['X'], 'Out', check_prim=True, check_pir=True
        )


class TestProd8DOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.python_api = raw_reduce_prod
        self.public_python_api = raw_reduce_prod
        self.init_data_type()
        self.init_inputs_and_outputs()

    def init_inputs_and_outputs(self):
        self.inputs = {
            'X': np.random.random((2, 5, 3, 2, 2, 3, 4, 2)).astype(
                self.data_type
            )
        }
        self.attrs = {'dim': [2, 3, 4]}
        self.outputs = {
            'Out': self.inputs['X'].prod(axis=tuple(self.attrs['dim']))
        }

    def init_data_type(self):
        self.data_type = (
            "float32" if core.is_compiled_with_rocm() else "float64"
        )

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)


@unittest.skipIf(
    not paddle.is_compiled_with_cuda(), "FP16 test runs only on GPU"
)
class TestProd8DFP16OP(TestProd8DOp):
    def init_data_type(self):
        self.data_type = "float16"

    def test_check_output(self):
        self.check_output_with_place(place=paddle.CUDAPlace(0), check_pir=True)

    def test_check_grad(self):
        self.check_grad_with_place(
            paddle.CUDAPlace(0), ['X'], 'Out', check_pir=True
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or paddle.is_compiled_with_rocm()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestProd8DBFP16OP(TestProd8DOp):
    def init_data_type(self):
        self.data_type = np.uint16

    def init_inputs_and_outputs(self):
        x = np.random.random((2, 5, 3, 2, 2, 3, 4, 2)).astype("float32")
        self.attrs = {'dim': [2, 3, 4]}
        out = x.prod(axis=tuple(self.attrs['dim']))
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def test_check_output(self):
        self.check_output_with_place(place=paddle.CUDAPlace(0), check_pir=True)

    def test_check_grad(self):
        self.check_grad_with_place(
            paddle.CUDAPlace(0), ['X'], 'Out', check_pir=True
        )


def reduce_all_wrapper(x, axis=None, keepdim=False, reduce_all=True, name=None):
    return paddle.all(x, axis, keepdim, name)


class TestAllOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_all"
        self.python_api = reduce_all_wrapper
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.outputs = {'Out': self.inputs['X'].all()}
        self.attrs = {'reduce_all': True}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAllFloatOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_all"
        self.python_api = reduce_all_wrapper
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("float")}
        self.outputs = {'Out': self.inputs['X'].all()}
        self.attrs = {'reduce_all': True}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAllIntOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_all"
        self.python_api = reduce_all_wrapper
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("int")}
        self.outputs = {'Out': self.inputs['X'].all()}
        self.attrs = {'reduce_all': True}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAllOp_ZeroDim(OpTest):
    def setUp(self):
        self.python_api = paddle.all
        self.op_type = "reduce_all"
        self.inputs = {'X': np.random.randint(0, 2, []).astype("bool")}
        self.outputs = {'Out': self.inputs['X'].all()}
        self.attrs = {'dim': []}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAll8DOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {
            'X': np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype(
                "bool"
            )
        }
        self.attrs = {'dim': (2, 3, 4)}
        self.outputs = {'Out': self.inputs['X'].all(axis=self.attrs['dim'])}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAllOpWithDim(OpTest):
    def setUp(self):
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.attrs = {'dim': (1,)}
        self.outputs = {'Out': self.inputs['X'].all(axis=self.attrs['dim'])}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAll8DOpWithDim(OpTest):
    def setUp(self):
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {
            'X': np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype(
                "bool"
            )
        }
        self.attrs = {'dim': (1, 3, 4)}
        self.outputs = {'Out': self.inputs['X'].all(axis=self.attrs['dim'])}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAllOpWithKeepDim(OpTest):
    def setUp(self):
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.attrs = {'dim': [1], 'keep_dim': True}
        self.outputs = {
            'Out': np.expand_dims(self.inputs['X'].all(axis=1), axis=1)
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAll8DOpWithKeepDim(OpTest):
    def setUp(self):
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {
            'X': np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype(
                "bool"
            )
        }
        self.attrs = {'dim': (5,), 'keep_dim': True}
        self.outputs = {
            'Out': np.expand_dims(
                self.inputs['X'].all(axis=self.attrs['dim']), axis=5
            )
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAllComplex64Op(OpTest):
    def setUp(self):
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        real_part = np.random.uniform(-1, 1, (2, 5, 3, 2, 2, 3, 4, 2))
        imag_part = np.random.uniform(-1, 1, (2, 5, 3, 2, 2, 3, 4, 2))
        self.inputs = {'X': (real_part + 1j * imag_part).astype("complex64")}
        self.attrs = {'dim': (5,), 'keep_dim': True}
        self.outputs = {
            'Out': np.expand_dims(
                self.inputs['X'].all(axis=self.attrs['dim']), axis=5
            )
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAllComplex64OpInf(TestAllComplex64Op):
    def setUp(self):
        super().setUp()
        real_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), np.inf)
        imag_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), np.inf)
        self.inputs['X'] = (real_part + 1j * imag_part).astype("complex64")
        self.outputs['Out'] = np.expand_dims(
            np.all(self.inputs['X'], axis=self.attrs['dim']), axis=5
        )


class TestAllComplex64OpNegInf(TestAllComplex64Op):
    def setUp(self):
        super().setUp()
        real_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), -np.inf)
        imag_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), -np.inf)
        self.inputs['X'] = (real_part + 1j * imag_part).astype("complex64")
        self.outputs['Out'] = np.expand_dims(
            np.all(self.inputs['X'], axis=self.attrs['dim']), axis=5
        )


class TestAllComplex64OpNan(TestAllComplex64Op):
    def setUp(self):
        super().setUp()
        real_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), np.nan)
        imag_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), np.nan)
        self.inputs['X'] = (real_part + 1j * imag_part).astype("complex64")
        self.outputs['Out'] = np.expand_dims(
            np.all(self.inputs['X'], axis=self.attrs['dim']), axis=5
        )


class TestAllComplex64OpZero(TestAllComplex64Op):
    def setUp(self):
        super().setUp()
        real_part = np.zeros((2, 5, 3, 2, 2, 3, 4, 2))
        imag_part = np.zeros((2, 5, 3, 2, 2, 3, 4, 2))
        self.inputs['X'] = (real_part + 1j * imag_part).astype("complex64")
        self.outputs['Out'] = np.expand_dims(
            np.all(self.inputs['X'], axis=self.attrs['dim']), axis=5
        )


class TestAllComplex64OpMixed(TestAllComplex64Op):
    def setUp(self):
        super().setUp()
        special_values = np.array(
            [np.inf, -np.inf, np.nan, 0], dtype=np.float64
        )
        real_part = np.random.choice(special_values, (2, 5, 3, 2, 2, 3, 4, 2))
        imag_part = np.random.choice(special_values, (2, 5, 3, 2, 2, 3, 4, 2))
        self.inputs['X'] = (real_part + 1j * imag_part).astype("complex64")
        self.outputs['Out'] = np.expand_dims(
            np.all(self.inputs['X'], axis=self.attrs['dim']), axis=5
        )


class TestAllComplex128Op(OpTest):
    def setUp(self):
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        real_part = np.random.uniform(-1, 1, (2, 5, 3, 2, 2, 3, 4, 2))
        imag_part = np.random.uniform(-1, 1, (2, 5, 3, 2, 2, 3, 4, 2))
        self.inputs = {'X': (real_part + 1j * imag_part).astype("complex128")}
        self.attrs = {'dim': (5,), 'keep_dim': True}
        self.outputs = {
            'Out': np.expand_dims(
                self.inputs['X'].all(axis=self.attrs['dim']), axis=5
            )
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAllComplex128OpInf(TestAllComplex128Op):
    def setUp(self):
        super().setUp()
        real_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), np.inf)
        imag_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), np.inf)
        self.inputs['X'] = (real_part + 1j * imag_part).astype("complex128")
        self.outputs['Out'] = np.expand_dims(
            np.all(self.inputs['X'], axis=self.attrs['dim']), axis=5
        )


class TestAllComplex128OpNegInf(TestAllComplex128Op):
    def setUp(self):
        super().setUp()
        real_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), -np.inf)
        imag_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), -np.inf)
        self.inputs['X'] = (real_part + 1j * imag_part).astype("complex128")
        self.outputs['Out'] = np.expand_dims(
            np.all(self.inputs['X'], axis=self.attrs['dim']), axis=5
        )


class TestAllComplex128OpNan(TestAllComplex128Op):
    def setUp(self):
        super().setUp()
        real_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), np.nan)
        imag_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), np.nan)
        self.inputs['X'] = (real_part + 1j * imag_part).astype("complex128")
        self.outputs['Out'] = np.expand_dims(
            np.all(self.inputs['X'], axis=self.attrs['dim']), axis=5
        )


class TestAllComplex128OpZero(TestAllComplex128Op):
    def setUp(self):
        super().setUp()
        real_part = np.zeros((2, 5, 3, 2, 2, 3, 4, 2))
        imag_part = np.zeros((2, 5, 3, 2, 2, 3, 4, 2))
        self.inputs['X'] = (real_part + 1j * imag_part).astype("complex128")
        self.outputs['Out'] = np.expand_dims(
            np.all(self.inputs['X'], axis=self.attrs['dim']), axis=5
        )


class TestAllComplex128OpMixed(TestAllComplex128Op):
    def setUp(self):
        super().setUp()
        special_values = np.array(
            [np.inf, -np.inf, np.nan, 0], dtype=np.float64
        )
        real_part = np.random.choice(special_values, (2, 5, 3, 2, 2, 3, 4, 2))
        imag_part = np.random.choice(special_values, (2, 5, 3, 2, 2, 3, 4, 2))
        self.inputs['X'] = (real_part + 1j * imag_part).astype("complex128")
        self.outputs['Out'] = np.expand_dims(
            np.all(self.inputs['X'], axis=self.attrs['dim']), axis=5
        )


class TestAllOpError(unittest.TestCase):

    def test_errors(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # The input type of reduce_all_op must be Variable.
            input1 = 12
            self.assertRaises(TypeError, paddle.all, input1)


def reduce_any_wrapper(x, axis=None, keepdim=False, reduce_all=True, name=None):
    return paddle.any(x, axis, keepdim, name)


class TestAnyOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_any"
        self.prim_op_type = "comp"
        self.python_api = reduce_any_wrapper
        self.public_python_api = reduce_any_wrapper
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.outputs = {'Out': self.inputs['X'].any()}
        self.attrs = {'reduce_all': True}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)


class TestAnyFloatOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_any"
        self.prim_op_type = "comp"
        self.python_api = reduce_any_wrapper
        self.public_python_api = reduce_any_wrapper
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("float")}
        self.outputs = {'Out': self.inputs['X'].any()}
        self.attrs = {'reduce_all': True}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)


class TestAnyIntOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_any"
        self.prim_op_type = "comp"
        self.python_api = reduce_any_wrapper
        self.public_python_api = reduce_any_wrapper
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("int")}
        self.outputs = {'Out': self.inputs['X'].any()}
        self.attrs = {'reduce_all': True}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)


class TestAnyComplex64Op(OpTest):
    def setUp(self):
        self.op_type = "reduce_any"
        self.python_api = paddle.any
        real_part = np.random.uniform(-1, 1, (2, 5, 3, 2, 2, 3, 4, 2))
        imag_part = np.random.uniform(-1, 1, (2, 5, 3, 2, 2, 3, 4, 2))
        self.inputs = {'X': (real_part + 1j * imag_part).astype("complex64")}
        self.attrs = {'dim': (5,), 'keep_dim': True}
        self.outputs = {
            'Out': np.expand_dims(
                self.inputs['X'].all(axis=self.attrs['dim']), axis=5
            )
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAnyComplex64OpInf(TestAnyComplex64Op):
    def setUp(self):
        super().setUp()
        real_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), np.inf)
        imag_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), np.inf)
        self.inputs['X'] = (real_part + 1j * imag_part).astype("complex64")
        self.outputs['Out'] = np.expand_dims(
            np.all(self.inputs['X'], axis=self.attrs['dim']), axis=5
        )


class TestAnyComplex64OpNegInf(TestAnyComplex64Op):
    def setUp(self):
        super().setUp()
        real_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), -np.inf)
        imag_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), -np.inf)
        self.inputs['X'] = (real_part + 1j * imag_part).astype("complex64")
        self.outputs['Out'] = np.expand_dims(
            np.all(self.inputs['X'], axis=self.attrs['dim']), axis=5
        )


class TestAnyComplex64OpNan(TestAnyComplex64Op):
    def setUp(self):
        super().setUp()
        real_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), np.nan)
        imag_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), np.nan)
        self.inputs['X'] = (real_part + 1j * imag_part).astype("complex64")
        self.outputs['Out'] = np.expand_dims(
            np.all(self.inputs['X'], axis=self.attrs['dim']), axis=5
        )


class TestAnyComplex64OpZero(TestAnyComplex64Op):
    def setUp(self):
        super().setUp()
        real_part = np.zeros((2, 5, 3, 2, 2, 3, 4, 2))
        imag_part = np.zeros((2, 5, 3, 2, 2, 3, 4, 2))
        self.inputs['X'] = (real_part + 1j * imag_part).astype("complex64")
        self.outputs['Out'] = np.expand_dims(
            np.all(self.inputs['X'], axis=self.attrs['dim']), axis=5
        )


class TestAnyComplex128Op(OpTest):
    def setUp(self):
        self.op_type = "reduce_any"
        self.python_api = paddle.any
        real_part = np.random.uniform(-1, 1, (2, 5, 3, 2, 2, 3, 4, 2))
        imag_part = np.random.uniform(-1, 1, (2, 5, 3, 2, 2, 3, 4, 2))
        self.inputs = {'X': (real_part + 1j * imag_part).astype("complex128")}
        self.attrs = {'dim': (5,), 'keep_dim': True}
        self.outputs = {
            'Out': np.expand_dims(
                self.inputs['X'].all(axis=self.attrs['dim']), axis=5
            )
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAnyComplex128OpInf(TestAnyComplex128Op):
    def setUp(self):
        super().setUp()
        real_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), np.inf)
        imag_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), np.inf)
        self.inputs['X'] = (real_part + 1j * imag_part).astype("complex128")
        self.outputs['Out'] = np.expand_dims(
            np.all(self.inputs['X'], axis=self.attrs['dim']), axis=5
        )


class TestAnyComplex128OpNegInf(TestAnyComplex128Op):
    def setUp(self):
        super().setUp()
        real_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), -np.inf)
        imag_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), -np.inf)
        self.inputs['X'] = (real_part + 1j * imag_part).astype("complex128")
        self.outputs['Out'] = np.expand_dims(
            np.all(self.inputs['X'], axis=self.attrs['dim']), axis=5
        )


class TestAnyComplex128OpNan(TestAnyComplex128Op):
    def setUp(self):
        super().setUp()
        real_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), np.nan)
        imag_part = np.full((2, 5, 3, 2, 2, 3, 4, 2), np.nan)
        self.inputs['X'] = (real_part + 1j * imag_part).astype("complex128")
        self.outputs['Out'] = np.expand_dims(
            np.all(self.inputs['X'], axis=self.attrs['dim']), axis=5
        )


class TestAnyComplex128OpZero(TestAnyComplex128Op):
    def setUp(self):
        super().setUp()
        real_part = np.zeros((2, 5, 3, 2, 2, 3, 4, 2))
        imag_part = np.zeros((2, 5, 3, 2, 2, 3, 4, 2))
        self.inputs['X'] = (real_part + 1j * imag_part).astype("complex128")
        self.outputs['Out'] = np.expand_dims(
            np.all(self.inputs['X'], axis=self.attrs['dim']), axis=5
        )


class TestAnyOp_ZeroDim(OpTest):
    def setUp(self):
        self.op_type = "reduce_any"
        self.prim_op_type = "comp"
        self.python_api = paddle.any
        self.public_python_api = paddle.any
        self.inputs = {'X': np.random.randint(0, 2, []).astype("bool")}
        self.outputs = {'Out': self.inputs['X'].any()}
        self.attrs = {'dim': []}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)


class TestAny8DOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_any"
        self.prim_op_type = "comp"
        self.python_api = paddle.any
        self.public_python_api = paddle.any
        self.inputs = {
            'X': np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype(
                "bool"
            )
        }
        self.attrs = {'dim': (3, 5, 4)}
        self.outputs = {'Out': self.inputs['X'].any(axis=self.attrs['dim'])}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)


class TestAnyOpWithDim(OpTest):
    def setUp(self):
        self.op_type = "reduce_any"
        self.prim_op_type = "comp"
        self.python_api = paddle.any
        self.public_python_api = paddle.any
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.attrs = {'dim': [1]}
        self.outputs = {'Out': self.inputs['X'].any(axis=1)}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)


class TestAny8DOpWithDim(OpTest):
    def setUp(self):
        self.op_type = "reduce_any"
        self.prim_op_type = "comp"
        self.python_api = paddle.any
        self.public_python_api = paddle.any
        self.inputs = {
            'X': np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype(
                "bool"
            )
        }
        self.attrs = {'dim': (3, 6)}
        self.outputs = {'Out': self.inputs['X'].any(axis=self.attrs['dim'])}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)


class TestAnyOpWithKeepDim(OpTest):
    def setUp(self):
        self.op_type = "reduce_any"
        self.prim_op_type = "comp"
        self.python_api = paddle.any
        self.public_python_api = paddle.any
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.attrs = {'dim': (1,), 'keep_dim': True}
        self.outputs = {
            'Out': np.expand_dims(
                self.inputs['X'].any(axis=self.attrs['dim']), axis=1
            )
        }

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)


class TestAny8DOpWithKeepDim(OpTest):
    def setUp(self):
        self.op_type = "reduce_any"
        self.prim_op_type = "comp"
        self.python_api = paddle.any
        self.public_python_api = paddle.any
        self.inputs = {
            'X': np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype(
                "bool"
            )
        }
        self.attrs = {'dim': (1,), 'keep_dim': True}
        self.outputs = {
            'Out': np.expand_dims(
                self.inputs['X'].any(axis=self.attrs['dim']), axis=1
            )
        }

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)


class TestAnyOpError(unittest.TestCase):

    def test_errors(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # The input type of reduce_any_op must be Variable.
            input1 = 12
            self.assertRaises(TypeError, paddle.any, input1)


class Test1DReduce(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = paddle.sum
        self.public_python_api = paddle.sum
        self.prim_op_type = "prim"
        self.inputs = {'X': np.random.random(120).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}
        self.if_enable_cinn()

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim=True)


class TestReduceSum_ZeroDim(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = paddle.sum
        self.public_python_api = paddle.sum
        self.prim_op_type = "prim"
        self.inputs = {'X': np.random.random(()).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}
        self.if_enable_cinn()


class Test2DReduce0(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = paddle.sum
        self.public_python_api = paddle.sum
        self.prim_op_type = "prim"
        self.attrs = {'dim': [0]}
        self.inputs = {'X': np.random.random((20, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}
        self.if_enable_cinn()


class Test2DReduce1(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = paddle.sum
        self.public_python_api = paddle.sum
        self.prim_op_type = "prim"
        self.attrs = {'dim': [1]}
        self.inputs = {'X': np.random.random((20, 10)).astype("float64")}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }
        self.if_enable_cinn()


class Test3DReduce0(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = paddle.sum
        self.public_python_api = paddle.sum
        self.prim_op_type = "prim"
        self.attrs = {'dim': [1]}
        self.inputs = {'X': np.random.random((5, 6, 7)).astype("float64")}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }
        self.if_enable_cinn()


class Test3DReduce1(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = paddle.sum
        self.public_python_api = paddle.sum
        self.prim_op_type = "prim"
        self.attrs = {'dim': [2]}
        self.inputs = {'X': np.random.random((5, 6, 7)).astype("float64")}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }
        self.if_enable_cinn()


class Test3DReduce2(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = paddle.sum
        self.public_python_api = paddle.sum
        self.prim_op_type = "prim"
        self.attrs = {'dim': [-2]}
        self.inputs = {'X': np.random.random((5, 6, 7)).astype("float64")}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }
        self.if_enable_cinn()


class Test3DReduce3(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = paddle.sum
        self.public_python_api = paddle.sum
        self.prim_op_type = "prim"
        self.attrs = {'dim': [1, 2]}
        self.inputs = {'X': np.random.random((5, 6, 7)).astype("float64")}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }
        self.if_enable_cinn()


def reduce_sum_wrapper2(x, axis=[0], dtype=None, keepdim=False):
    if paddle.in_dynamic_mode():
        return paddle._C_ops.sum(x, axis, dtype, keepdim)
    else:
        if in_pir_mode():
            return paddle._pir_ops.sum(x, axis, dtype, keepdim)


class Test8DReduce0(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = reduce_sum_wrapper2
        self.attrs = {'dim': (4, 2, 3)}
        self.inputs = {
            'X': np.random.random((2, 5, 3, 2, 2, 3, 4, 2)).astype("float64")
        }
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestKeepDimReduce(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = paddle.sum
        self.public_python_api = paddle.sum
        self.prim_op_type = "prim"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [1], 'keep_dim': True}
        self.outputs = {
            'Out': self.inputs['X'].sum(
                axis=tuple(self.attrs['dim']), keepdims=self.attrs['keep_dim']
            )
        }
        self.if_enable_cinn()


class TestKeepDimReduceForEager(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = reduce_sum_wrapper2
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [1], 'keep_dim': True}
        self.outputs = {
            'Out': self.inputs['X'].sum(
                axis=tuple(self.attrs['dim']), keepdims=self.attrs['keep_dim']
            )
        }

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestKeepDim8DReduce(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = reduce_sum_wrapper2
        self.inputs = {
            'X': np.random.random((2, 5, 3, 2, 2, 3, 4, 2)).astype("float64")
        }
        self.attrs = {'dim': (3, 4, 5), 'keep_dim': True}
        self.outputs = {
            'Out': self.inputs['X'].sum(
                axis=tuple(self.attrs['dim']), keepdims=self.attrs['keep_dim']
            )
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework."
)
class TestReduceMaxOpMultiAxes(OpTest):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.prim_op_type = "prim"
        self.python_api = paddle.max
        self.public_python_api = paddle.max
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [-2, -1]}
        self.outputs = {
            'Out': self.inputs['X'].max(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        # only composite op support gradient check of reduce_max
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            only_check_prim=True,
            check_pir=True,
        )


@skip_check_grad_ci(
    reason="reduce_min is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework."
)
class TestReduceMinOpMultiAxes(OpTest):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.python_api = paddle.min
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [1, 2]}
        self.outputs = {
            'Out': self.inputs['X'].min(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output()


class TestKeepDimReduceSumMultiAxes(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = paddle.sum
        self.public_python_api = paddle.sum
        self.prim_op_type = "prim"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [-2, -1], 'keep_dim': True}
        self.outputs = {
            'Out': self.inputs['X'].sum(
                axis=tuple(self.attrs['dim']), keepdims=True
            )
        }
        self.if_enable_cinn()

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim=True)


class TestKeepDimReduceSumMultiAxesForEager(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = reduce_sum_wrapper2
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [-2, -1], 'keep_dim': True}
        self.outputs = {
            'Out': self.inputs['X'].sum(
                axis=tuple(self.attrs['dim']), keepdims=True
            )
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestReduceSumWithDimOne(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = paddle.sum
        self.public_python_api = paddle.sum
        self.prim_op_type = "prim"
        self.inputs = {'X': np.random.random((100, 1, 1)).astype("float64")}
        self.attrs = {'dim': [1, 2], 'keep_dim': True}
        self.outputs = {
            'Out': self.inputs['X'].sum(
                axis=tuple(self.attrs['dim']), keepdims=True
            )
        }
        self.if_enable_cinn()

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim=True)


class TestReduceSumWithDimOneForEager(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = reduce_sum_wrapper2
        self.inputs = {'X': np.random.random((100, 1, 1)).astype("float64")}
        self.attrs = {'dim': [1, 2], 'keep_dim': True}
        self.outputs = {
            'Out': self.inputs['X'].sum(
                axis=tuple(self.attrs['dim']), keepdims=True
            )
        }
        self.enable_cinn = True

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestReduceSumWithNumelOne(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = paddle.sum
        self.public_python_api = paddle.sum
        self.prim_op_type = "prim"
        self.inputs = {'X': np.random.random((100, 1)).astype("float64")}
        self.attrs = {'dim': [1], 'keep_dim': False}
        self.outputs = {
            'Out': self.inputs['X'].sum(
                axis=tuple(self.attrs['dim']), keepdims=False
            )
        }
        self.if_enable_cinn()

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim=False)


def reduce_sum_wrapper(
    x, axis=None, keepdim=False, reduce_all=True, out_dtype=None, name=None
):
    return paddle.sum(x, axis, out_dtype, keepdim, name)


class TestReduceAll(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = reduce_sum_wrapper
        self.public_python_api = reduce_sum_wrapper
        self.prim_op_type = "prim"
        self.inputs = {'X': np.random.random((100, 1, 1)).astype("float64")}
        self.attrs = {'reduce_all': True, 'keep_dim': False}
        self.outputs = {'Out': self.inputs['X'].sum()}
        self.if_enable_cinn()

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim=True)


class TestReduceAllFp32(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = reduce_sum_wrapper
        self.public_python_api = reduce_sum_wrapper
        self.prim_op_type = "prim"
        self.inputs = {'X': np.random.random((100, 1, 1)).astype("float32")}
        self.attrs = {'reduce_all': True, 'keep_dim': False}
        self.outputs = {'Out': self.inputs['X'].sum()}
        self.if_enable_cinn()

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim=True)


class Test1DReduceWithAxes1(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = paddle.sum
        self.public_python_api = paddle.sum
        self.prim_op_type = "prim"
        self.inputs = {'X': np.random.random(100).astype("float64")}
        self.attrs = {'dim': [0], 'keep_dim': False}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}
        self.if_enable_cinn()

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim=True)


def reduce_sum_wrapper_fp64(
    x, axis=None, keepdim=False, reduce_all=True, out_dtype=None, name=None
):
    return paddle.sum(x, axis, 'float64', keepdim, name)


class TestReduceWithDtype(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = reduce_sum_wrapper_fp64
        self.public_python_api = reduce_sum_wrapper_fp64
        self.prim_op_type = "prim"
        self.inputs = {'X': np.random.random((6, 2, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].sum().astype('float64')}
        self.attrs = {'reduce_all': True}
        self.attrs.update(
            {
                'in_dtype': paddle.float32,
                'out_dtype': paddle.float64,
            }
        )
        self.if_enable_cinn()

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim=True)


class TestReduceWithDtype1(TestReduceWithDtype):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = paddle.sum
        self.public_python_api = paddle.sum
        self.prim_op_type = "prim"
        self.inputs = {'X': np.random.random((6, 2, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=1)}
        self.attrs = {'dim': [1]}
        self.attrs.update(
            {
                'in_dtype': paddle.float32,
                'out_dtype': paddle.float64,
            }
        )
        # cinn op_mapper not support in_dtype/out_dtype attr
        self.enable_cinn = False

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim=True)


class TestReduceWithDtype2(TestReduceWithDtype):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.prim_op_type = "prim"
        self.python_api = paddle.sum
        self.public_python_api = paddle.sum
        self.inputs = {'X': np.random.random((6, 2, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=1, keepdims=True)}
        self.attrs = {'dim': [1], 'keep_dim': True}
        self.attrs.update(
            {
                'in_dtype': paddle.float32,
                'out_dtype': paddle.float64,
            }
        )
        # cinn op_mapper not support in_dtype/out_dtype attr
        self.enable_cinn = False

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim=True)


class TestReduceSumOpError(unittest.TestCase):
    def test_errors1(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                # The input type of reduce_sum_op must be Variable.
                x1 = base.create_lod_tensor(
                    np.array([[-1]]), [[1]], base.CPUPlace()
                )
                self.assertRaises(TypeError, paddle.sum, x1)
                # The input dtype of reduce_sum_op  must be float32 or float64 or int32 or int64.


class API_TestSumOp(unittest.TestCase):
    def run_static(
        self, shape, x_dtype, attr_axis, attr_dtype=None, np_axis=None
    ):
        if np_axis is None:
            np_axis = attr_axis

        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for place in places:
            with base.program_guard(base.Program(), base.Program()):
                data = paddle.static.data("data", shape=shape, dtype=x_dtype)
                result_sum = paddle.sum(
                    x=data, axis=attr_axis, dtype=attr_dtype
                )

                exe = base.Executor(place)
                input_data = np.random.rand(*shape).astype(x_dtype)
                (res,) = exe.run(
                    feed={"data": input_data}, fetch_list=[result_sum]
                )

            np.testing.assert_allclose(
                res,
                np.sum(input_data.astype(attr_dtype), axis=np_axis),
                rtol=1e-05,
            )

    def test_static(self):
        shape = [10, 10]
        axis = 1

        self.run_static(shape, "bool", axis, attr_dtype=None)
        self.run_static(shape, "bool", axis, attr_dtype="int32")
        self.run_static(shape, "bool", axis, attr_dtype="int64")
        self.run_static(shape, "bool", axis, attr_dtype="float16")

        self.run_static(shape, "int32", axis, attr_dtype=None)
        self.run_static(shape, "int32", axis, attr_dtype="int32")
        self.run_static(shape, "int32", axis, attr_dtype="int64")
        self.run_static(shape, "int32", axis, attr_dtype="float64")

        self.run_static(shape, "int64", axis, attr_dtype=None)
        self.run_static(shape, "int64", axis, attr_dtype="int64")
        self.run_static(shape, "int64", axis, attr_dtype="int32")

        self.run_static(shape, "float32", axis, attr_dtype=None)
        self.run_static(shape, "float32", axis, attr_dtype="float32")
        self.run_static(shape, "float32", axis, attr_dtype="float64")
        self.run_static(shape, "float32", axis, attr_dtype="int64")

        self.run_static(shape, "float64", axis, attr_dtype=None)
        self.run_static(shape, "float64", axis, attr_dtype="float32")
        self.run_static(shape, "float64", axis, attr_dtype="float64")

        shape = [5, 5, 5]
        self.run_static(shape, "int32", (0, 1), attr_dtype="int32")
        self.run_static(
            shape, "int32", (), attr_dtype="int32", np_axis=(0, 1, 2)
        )

    def test_dygraph(self):
        np_x = np.random.random([2, 3, 4]).astype('int32')
        with base.dygraph.guard():
            x = paddle.to_tensor(np_x)
            out0 = paddle.sum(x).numpy()
            out1 = paddle.sum(x, axis=0).numpy()
            out2 = paddle.sum(x, axis=(0, 1)).numpy()
            out3 = paddle.sum(x, axis=(0, 1, 2)).numpy()

        self.assertTrue((out0 == np.sum(np_x, axis=(0, 1, 2))).all())
        self.assertTrue((out1 == np.sum(np_x, axis=0)).all())
        self.assertTrue((out2 == np.sum(np_x, axis=(0, 1))).all())
        self.assertTrue((out3 == np.sum(np_x, axis=(0, 1, 2))).all())


class TestAllAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_static_result(self, place):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with base.program_guard(main, startup):
            input = paddle.static.data(name="input", shape=[4, 4], dtype="bool")
            result = paddle.all(x=input)
            input_np = np.random.randint(0, 2, [4, 4]).astype("bool")

            exe = base.Executor(place)
            fetches = exe.run(
                main,
                feed={"input": input_np},
                fetch_list=[result],
            )
            self.assertTrue((fetches[0] == np.all(input_np)).all())

    def check_static_float_result(self, place):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with base.program_guard(main, startup):
            input = paddle.static.data(
                name="input", shape=[4, 4], dtype="float"
            )
            result = paddle.all(x=input)
            input_np = np.random.randint(0, 2, [4, 4]).astype("float")

            exe = base.Executor(place)
            fetches = exe.run(
                main,
                feed={"input": input_np},
                fetch_list=[result],
            )
            self.assertTrue((fetches[0] == np.all(input_np)).all())

    def check_static_int_result(self, place):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with base.program_guard(main, startup):
            input = paddle.static.data(name="input", shape=[4, 4], dtype="int")
            result = paddle.all(x=input)
            input_np = np.random.randint(0, 2, [4, 4]).astype("int")

            exe = base.Executor(place)
            fetches = exe.run(
                main,
                feed={"input": input_np},
                fetch_list=[result],
            )
            self.assertTrue((fetches[0] == np.all(input_np)).all())

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)
            self.check_static_float_result(place=place)
            self.check_static_int_result(place=place)

    def test_dygraph(self):
        paddle.disable_static()
        for place in self.places:
            with base.dygraph.guard(place):
                np_x = np.random.randint(0, 2, (12, 10)).astype(np.bool_)
                x = paddle.assign(np_x)
                x = paddle.cast(x, 'bool')

                out1 = paddle.all(x)
                np_out1 = out1.numpy()
                expect_res1 = np.all(np_x)
                self.assertTrue((np_out1 == expect_res1).all())

                out2 = paddle.all(x, axis=0)
                np_out2 = out2.numpy()
                expect_res2 = np.all(np_x, axis=0)
                self.assertTrue((np_out2 == expect_res2).all())

                out3 = paddle.all(x, axis=-1)
                np_out3 = out3.numpy()
                expect_res3 = np.all(np_x, axis=-1)
                self.assertTrue((np_out3 == expect_res3).all())

                out4 = paddle.all(x, axis=1, keepdim=True)
                np_out4 = out4.numpy()
                expect_res4 = np.all(np_x, axis=1, keepdims=True)
                self.assertTrue((np_out4 == expect_res4).all())

                x = paddle.cast(x, 'float')
                out5 = paddle.all(x)
                np_out5 = out5.numpy()
                expect_res5 = np.all(np_x)
                self.assertTrue((np_out5 == expect_res5).all())

                x = paddle.cast(x, 'int')
                out6 = paddle.all(x)
                np_out6 = out6.numpy()
                expect_res6 = np.all(np_x)
                self.assertTrue((np_out6 == expect_res6).all())

        paddle.enable_static()


class TestAnyAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_static_result(self, place):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with base.program_guard(main, startup):
            input = paddle.static.data(name="input", shape=[4, 4], dtype="bool")
            result = paddle.any(x=input)
            input_np = np.random.randint(0, 2, [4, 4]).astype("bool")

            exe = base.Executor(place)
            fetches = exe.run(
                main,
                feed={"input": input_np},
                fetch_list=[result],
            )
            self.assertTrue((fetches[0] == np.any(input_np)).all())

    def check_static_float_result(self, place):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with base.program_guard(main, startup):
            input = paddle.static.data(
                name="input", shape=[4, 4], dtype="float"
            )
            result = paddle.any(x=input)
            input_np = np.random.randint(0, 2, [4, 4]).astype("float")

            exe = base.Executor(place)
            fetches = exe.run(
                main,
                feed={"input": input_np},
                fetch_list=[result],
            )
            self.assertTrue((fetches[0] == np.any(input_np)).all())

    def check_static_int_result(self, place):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with base.program_guard(main, startup):
            input = paddle.static.data(name="input", shape=[4, 4], dtype="int")
            result = paddle.any(x=input)
            input_np = np.random.randint(0, 2, [4, 4]).astype("int")

            exe = base.Executor(place)
            fetches = exe.run(
                main,
                feed={"input": input_np},
                fetch_list=[result],
            )
            self.assertTrue((fetches[0] == np.any(input_np)).all())

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)
            self.check_static_float_result(place=place)
            self.check_static_int_result(place=place)

    def test_dygraph(self):
        paddle.disable_static()
        for place in self.places:
            with base.dygraph.guard(place):
                np_x = np.random.randint(0, 2, (12, 10)).astype(np.bool_)
                x = paddle.assign(np_x)
                x = paddle.cast(x, 'bool')

                out1 = paddle.any(x)
                np_out1 = out1.numpy()
                expect_res1 = np.any(np_x)
                self.assertTrue((np_out1 == expect_res1).all())

                out2 = paddle.any(x, axis=0)
                np_out2 = out2.numpy()
                expect_res2 = np.any(np_x, axis=0)
                self.assertTrue((np_out2 == expect_res2).all())

                out3 = paddle.any(x, axis=-1)
                np_out3 = out3.numpy()
                expect_res3 = np.any(np_x, axis=-1)
                self.assertTrue((np_out3 == expect_res3).all())

                out4 = paddle.any(x, axis=1, keepdim=True)
                np_out4 = out4.numpy()
                expect_res4 = np.any(np_x, axis=1, keepdims=True)
                self.assertTrue((np_out4 == expect_res4).all())

                np_x = np.random.randint(0, 2, (12, 10)).astype(np.float32)
                x = paddle.assign(np_x)
                x = paddle.cast(x, 'float32')

                out5 = paddle.any(x)
                np_out5 = out5.numpy()
                expect_res5 = np.any(np_x)
                self.assertTrue((np_out5 == expect_res5).all())

                x = paddle.cast(x, 'int')
                out6 = paddle.any(x)
                np_out6 = out6.numpy()
                expect_res6 = np.any(np_x)
                self.assertTrue((np_out6 == expect_res6).all())

        paddle.enable_static()


class TestAllZero(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.shape = [1, 0, 2]
        self.dtypes = [
            "bool",
            "float32",
            "float64",
            "int32",
            "complex64",
            "complex128",
        ]
        self.places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def calculate_expected_result(self, x_np, axis, keepdim):
        expected_result = np.all(x_np, axis=axis, keepdims=keepdim)
        return expected_result

    def check_result(
        self, static_result, expected_result, axis, keepdim, dtype, place
    ):
        self.assertTrue(
            (static_result == expected_result).all(),
            f"Static Mode - Shape: {self.shape}, Axis: {axis}, Keepdim: {keepdim}, Dtype: {dtype}, Place: {place}",
        )

    def _test_static(self, place, axis, keepdim, dtype):
        with static_guard():
            with base.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input = paddle.static.data(
                    name="x", shape=self.shape, dtype=dtype
                )
                result = paddle.all(x=input, axis=axis, keepdim=keepdim)
                x_np = np.zeros(self.shape, dtype=dtype)

                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"x": x_np},
                    fetch_list=[result],
                )
                expected_result = self.calculate_expected_result(
                    x_np, axis, keepdim
                )
                self.check_result(
                    fetches[0], expected_result, axis, keepdim, dtype, place
                )

    def _test_dygraph(self, place, axis, keepdim, dtype):
        with dygraph_guard():
            x_np = np.zeros(self.shape, dtype=dtype)
            x = paddle.to_tensor(x_np)
            dygraph_result = paddle.all(x, axis=axis, keepdim=keepdim).numpy()
            expected_result = self.calculate_expected_result(
                x_np, axis, keepdim
            )
            self.check_result(
                dygraph_result, expected_result, axis, keepdim, dtype, place
            )

    def _test_all(self, place, axis, keepdim, dtype):
        self._test_dygraph(place, axis, keepdim, dtype)
        self._test_static(place, axis, keepdim, dtype)

    def test_zero_size(self):
        axes_options = [
            None,
            0,
            1,
            2,
            -1,
            -2,
            (),
            (0, 1),
            (0, 2),
            (1, 2),
            (-1, -2),
        ]
        keepdims_options = [True, False]
        for place in self.places:
            for dtype in self.dtypes:
                for axis in axes_options:
                    for keepdim in keepdims_options:
                        self._test_all(place, axis, keepdim, dtype)


class TestAnyZero(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.shape = [1, 0, 2]
        self.dtypes = [
            "bool",
            "float32",
            "float64",
            "int32",
            "complex64",
            "complex128",
        ]
        self.places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def calculate_expected_result(self, x_np, axis, keepdim):
        expected_result = np.any(x_np, axis=axis, keepdims=keepdim)
        return expected_result

    def check_result(
        self, static_result, expected_result, axis, keepdim, dtype, place
    ):
        self.assertTrue(
            (static_result == expected_result).all(),
            f"Static Mode - Shape: {self.shape}, Axis: {axis}, Keepdim: {keepdim}, Dtype: {dtype}, Place: {place}",
        )

    def _test_static(self, place, axis, keepdim, dtype):
        with static_guard():
            with base.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input = paddle.static.data(
                    name="x", shape=self.shape, dtype=dtype
                )
                result = paddle.any(x=input, axis=axis, keepdim=keepdim)
                x_np = np.zeros(self.shape, dtype=dtype)

                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"x": x_np},
                    fetch_list=[result],
                )
                expected_result = self.calculate_expected_result(
                    x_np, axis, keepdim
                )
                self.check_result(
                    fetches[0], expected_result, axis, keepdim, dtype, place
                )

    def _test_dygraph(self, place, axis, keepdim, dtype):
        with dygraph_guard():
            x_np = np.zeros(self.shape, dtype=dtype)
            x = paddle.to_tensor(x_np)
            dygraph_result = paddle.any(x, axis=axis, keepdim=keepdim).numpy()
            expected_result = self.calculate_expected_result(
                x_np, axis, keepdim
            )
            self.check_result(
                dygraph_result, expected_result, axis, keepdim, dtype, place
            )

    def _test_any(self, place, axis, keepdim, dtype):
        self._test_dygraph(place, axis, keepdim, dtype)
        self._test_static(place, axis, keepdim, dtype)

    def test_zero_size(self):
        axes_options = [
            None,
            0,
            1,
            2,
            -1,
            -2,
            (),
            (0, 1),
            (0, 2),
            (1, 2),
            (-1, -2),
        ]
        keepdims_options = [True, False]
        for place in self.places:
            for dtype in self.dtypes:
                for axis in axes_options:
                    for keepdim in keepdims_options:
                        self._test_any(place, axis, keepdim, dtype)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
