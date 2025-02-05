# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
from test_adamw_op import adamw_step, adamw_wrapper

import paddle
from paddle.base import core


# test rsqrt, rsqrt_grad op
class TestRsqrt_FLoat16(OpTest):
    def setUp(self):
        self.op_type = "rsqrt"
        self.python_api = paddle.rsqrt
        self.public_python_api = paddle.rsqrt
        self.prim_op_type = "prim"
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = 1.0 / np.sqrt(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}

    def init_shape(self):
        self.shape = [10, 12]

    def init_dtype(self):
        self.dtype = np.float16

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        place = core.CUDAPlace(0)
        support_fp16 = core.is_float16_supported(place)
        if support_fp16:
            self.check_output_with_place(
                place,
                atol=1e-3,
                check_dygraph=True,
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        support_fp16 = core.is_float16_supported(place)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_dygraph=True,
            check_prim=True,
            check_prim_pir=True,
            max_relative_error=1e-2,
            check_pir=True,
        )


# test cos op
class TestCos_FLoat16(OpTest):
    def setUp(self):
        self.op_type = "cos"
        self.python_api = paddle.cos
        self.public_python_api = paddle.cos
        self.prim_op_type = "prim"
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = 1.0 / np.sqrt(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}

    def init_shape(self):
        self.shape = [10, 12]

    def init_dtype(self):
        self.dtype = np.float16

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        place = core.CUDAPlace(0)
        support_fp16 = core.is_float16_supported(place)
        if support_fp16:
            self.check_output_with_place(
                place,
                atol=1e-3,
                check_dygraph=True,
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        support_fp16 = core.is_float16_supported(place)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_dygraph=True,
            check_prim=True,
            check_prim_pir=True,
            max_relative_error=1e-2,
            check_pir=True,
        )


# test sin op
class TestSin_FLoat16(OpTest):
    def setUp(self):
        self.op_type = "sin"
        self.python_api = paddle.sin
        self.public_python_api = paddle.sin
        self.prim_op_type = "prim"
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = 1.0 / np.sqrt(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}

    def init_shape(self):
        self.shape = [10, 12]

    def init_dtype(self):
        self.dtype = np.float16

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        place = core.CUDAPlace(0)
        support_fp16 = core.is_float16_supported(place)
        if support_fp16:
            self.check_output_with_place(
                place,
                atol=1e-3,
                check_dygraph=True,
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        support_fp16 = core.is_float16_supported(place)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_dygraph=True,
            check_prim=True,
            check_prim_pir=True,
            max_relative_error=1e-2,
            check_pir=True,
        )


# test silu op
class TestSilu_FLoat16(OpTest):
    def setUp(self):
        self.op_type = "silu"
        self.python_api = paddle.nn.functional.silu
        self.public_python_api = paddle.nn.functional.silu
        self.prim_op_type = "prim"
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = 1.0 / np.sqrt(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}

    def init_shape(self):
        self.shape = [10, 12]

    def init_dtype(self):
        self.dtype = np.float16

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        place = core.CUDAPlace(0)
        support_fp16 = core.is_float16_supported(place)
        if support_fp16:
            self.check_output_with_place(
                place,
                atol=1e-3,
                check_dygraph=True,
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        support_fp16 = core.is_float16_supported(place)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_dygraph=True,
            check_prim=True,
            check_prim_pir=True,
            max_relative_error=1e-2,
            check_pir=True,
        )


# test sqrt op
class TestSqrt_FLoat16(OpTest):
    def setUp(self):
        self.op_type = "sqrt"
        self.python_api = paddle.sqrt
        self.public_python_api = paddle.sqrt
        self.prim_op_type = "prim"
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = 1.0 / np.sqrt(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}

    def init_shape(self):
        self.shape = [10, 12]

    def init_dtype(self):
        self.dtype = np.float16

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        place = core.CUDAPlace(0)
        support_fp16 = core.is_float16_supported(place)
        if support_fp16:
            self.check_output_with_place(
                place,
                atol=1e-3,
                check_dygraph=True,
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        support_fp16 = core.is_float16_supported(place)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_dygraph=True,
            check_prim=True,
            check_prim_pir=True,
            max_relative_error=1e-2,
            check_pir=True,
        )


# test divide op
class TestElementwiseDivFP16Op(OpTest):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.python_api = paddle.divide
        self.public_python_api = paddle.divide
        self.prim_op_type = "prim"
        self.init_args()
        self.init_dtype()
        self.init_shape()
        self.if_check_prim()
        self.if_enable_cinn()
        self.max_relative_error = 2e-3
        x = self.gen_data(self.x_shape).astype(self.val_dtype)
        y = self.gen_data(self.y_shape).astype(self.val_dtype)
        out = self.compute_output(x, y).astype(self.val_dtype)
        grad_out = np.ones(out.shape).astype(self.val_dtype)
        grad_x = self.compute_gradient_x(grad_out, y).astype(self.val_dtype)
        grad_y = self.compute_gradient_y(grad_out, out, y).astype(
            self.val_dtype
        )

        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}
        self.grad_out = grad_out
        self.grad_x = grad_x
        self.grad_y = grad_y

    def gen_data(self, shape):
        return np.random.uniform(0.1, 1, shape)

    def init_shape(self):
        self.x_shape = [13, 17]
        self.y_shape = [13, 17]

    def compute_output(self, x, y):
        return x / y

    def compute_gradient_x(self, grad_out, y):
        return grad_out / y

    def compute_gradient_y(self, grad_out, out, y):
        return -1 * grad_out * out / y

    def init_args(self):
        self.check_pir = True
        self.check_dygraph = True
        self.place = None

    def init_dtype(self):
        self.dtype = np.float16
        self.val_dtype = np.float16

    def if_enable_cinn(self):
        self.enable_cinn = True

    def if_check_prim(self):
        self.check_prim = True
        self.check_prim_pir = True

    def test_check_gradient(self):
        check_list = []
        check_list.append(
            {
                'grad': ['X', 'Y'],
                'no_grad': None,
                'val_grad': [self.grad_x, self.grad_y],
            }
        )
        check_list.append(
            {'grad': ['Y'], 'no_grad': set('X'), 'val_grad': [self.grad_y]}
        )
        check_list.append(
            {'grad': ['X'], 'no_grad': set('Y'), 'val_grad': [self.grad_x]}
        )
        for check_option in check_list:
            check_args = [check_option['grad'], 'Out']
            check_kwargs = {
                'no_grad_set': check_option['no_grad'],
                'user_defined_grads': check_option['val_grad'],
                'user_defined_grad_outputs': [self.grad_out],
                'check_dygraph': self.check_dygraph,
                'max_relative_error': self.max_relative_error,
            }
            if self.place is None:
                self.check_grad(*check_args, **check_kwargs, check_pir=True)
            else:
                check_args.insert(0, self.place)
                self.check_grad_with_place(
                    *check_args,
                    **check_kwargs,
                    check_pir=True,
                    check_prim=True,
                    check_prim_pir=True,
                )


# test mean op
class TestFP16MeanOp(OpTest):

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
        self.dtype = np.float16
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_output_with_place(place, check_pir=True)

    def test_checkout_grad(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            with paddle.base.dygraph.guard():
                x_np = np.random.random((10, 10)).astype(self.dtype)
                x = paddle.to_tensor(x_np)
                x.stop_gradient = False
                y = paddle.mean(x)
                dx = paddle.grad(y, x)[0].numpy()
                dx_expected = self.dtype(1.0 / np.prod(x_np.shape)) * np.ones(
                    x_np.shape
                ).astype(self.dtype)
                np.testing.assert_array_equal(dx, dx_expected)


# test scale op
class TestScaleFp16Op(OpTest):

    def setUp(self):
        self.op_type = "scale"
        self.python_api = paddle.scale
        self.init_dtype_type()
        self.public_python_api = paddle.scale
        self.prim_op_type = "prim"
        self.inputs = {'X': np.random.random((10, 10)).astype(self.dtype)}
        self.attrs = {'scale': -2.3}
        self.outputs = {
            'Out': self.inputs['X'] * self.dtype(self.attrs['scale'])
        }

    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(check_cinn=True, check_pir=True)

    def test_check_grad(self):
        self.check_grad(["X"], "Out", check_pir=True, check_prim_pir=True)


# test sum op
class TestAFP16SumOp(OpTest):

    def setUp(self):
        self.op_type = "sum"
        self.python_api = paddle.add_n
        self.public_python_api = paddle.add_n
        self.prim_op_type = "comp"
        self.init_kernel_type()
        self.use_mkldnn = False
        x0 = np.random.random((3, 40)).astype(self.dtype)
        x1 = np.random.random((3, 40)).astype(self.dtype)
        x2 = np.random.random((3, 40)).astype(self.dtype)
        self.inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
        y = x0 + x1 + x2
        self.outputs = {'Out': y}
        self.attrs = {'use_mkldnn': self.use_mkldnn}

    def init_kernel_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_output_with_place(
                place,
                check_cinn=True,
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
            )

    # FIXME: Because of the precision fp16, max_relative_error
    # should be 0.15 here.
    def test_check_grad(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad(
                ['x0'],
                'Out',
                check_cinn=True,
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
            )


# test any op
class TestAnyOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_any"
        self.prim_op_type = "comp"
        self.python_api = paddle.any
        self.public_python_api = paddle.any
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.outputs = {'Out': self.inputs['X'].any()}
        self.attrs = {'reduce_all': True}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)


# test add, add_grad op
class TestFP16ElementwiseAddOp(OpTest):

    def setUp(self):
        self.op_type = "elementwise_add"
        self.python_api = paddle.add
        self.public_python_api = paddle.add
        self.prim_op_type = "prim"
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()
        self.if_check_prim()
        self.if_enable_cinn()

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.out}

    def init_kernel_type(self):
        self.use_mkldnn = False

    def check_dygraph(self):
        return not self.use_mkldnn and self.axis == -1

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.add(self.x, self.y)

    def init_axis(self):
        self.axis = -1

    def if_check_prim(self):
        self.check_prim = self.axis == -1

    def if_enable_cinn(self):
        pass

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place,
            atol=1e-3,
            check_dygraph=self.check_dygraph(),
            check_pir=self.check_dygraph(),
        )

    def test_check_grad_normal(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X', 'Y'], 'Out', check_prim=True)

    def test_check_grad_ignore_x(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )

    def test_check_grad_ignore_y(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )


# test multiply, multiply_grad op
class TestElementwiseMulOpFp16(OpTest):

    def setUp(self):
        self.op_type = "elementwise_mul"
        self.prim_op_type = "prim"
        self.python_api = paddle.multiply
        self.public_python_api = paddle.multiply
        self.dtype = np.float64
        self.axis = -1
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()
        self.if_enable_cinn()

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.outputs = {'Out': self.out}
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}

    def init_kernel_type(self):
        self.use_mkldnn = False

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    def init_dtype(self):
        self.dtype = np.float16

    def init_axis(self):
        pass

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        self.check_output(
            check_dygraph=(not self.use_mkldnn),
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_grad_normal(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_dygraph=(not self.use_mkldnn),
            check_prim=True,
            check_prim_pir=(not self.use_mkldnn),
            check_pir=(not self.use_mkldnn),
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_grad_ignore_x(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            check_dygraph=(not self.use_mkldnn),
            check_prim=True,
            check_prim_pir=(not self.use_mkldnn),
            check_pir=(not self.use_mkldnn),
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_grad_ignore_y(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_dygraph=(not self.use_mkldnn),
            check_prim=True,
            check_prim_pir=(not self.use_mkldnn),
            check_pir=(not self.use_mkldnn),
            check_pir_onednn=self.check_pir_onednn,
        )


# test elementwise_pow op
class TestElementwisePowOpFP16(OpTest):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.dtype = np.float16
        self.python_api = paddle.pow
        self.public_python_api = paddle.pow
        self.prim_op_type = "prim"

        self.inputs = {
            'X': np.random.uniform(1, 2, [20, 5]).astype("float16"),
            'Y': np.random.uniform(1, 2, [20, 5]).astype("float16"),
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    def pow_grad(self, x, y, dout):
        dx = dout * y * np.power(x, (y - 1))
        dy = dout * np.log(x) * np.power(x, y)
        return dx, dy

    def test_check_output(self):
        if hasattr(self, 'attrs'):
            self.check_output(check_dygraph=False)
        else:
            self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            user_defined_grads=self.pow_grad(
                self.inputs['X'], self.inputs['Y'], 1 / self.inputs['X'].size
            ),
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )


# test adamw op
class TestAdamW(OpTest):
    def setUp(self):
        '''Test AdamW Op with supplied attributes'''
        self.op_type = "adamw"
        self.python_api = adamw_wrapper
        self.python_out_sig = ['Out']
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.004
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32"),
        }

        self.attrs = {
            'epsilon': epsilon,
            'beta1': beta1,
            'beta2': beta2,
            "coeff": 0.5,
            "with_decay": True,
        }

        param_out, moment1_out, moment2_out = adamw_step(
            self.inputs, self.attrs
        )

        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'ParamOut': param_out,
            'Beta1PowOut': np.array([beta1_pow]).astype("float32") * beta1,
            'Beta2PowOut': np.array([beta2_pow]).astype("float32") * beta2,
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


# test matmul, matmul_grad op
class TestMatMulV2Op(OpTest):

    def config(self):
        self.x_shape = (100,)
        self.y_shape = (100,)
        self.trans_x = False
        self.trans_y = False

    def init_kernel_type(self):
        self.dtype = np.float16

    def setUp(self):
        self.init_kernel_type()
        self.config()
        self.op_type = "matmul_v2"
        self.prim_op_type = "prim"
        self.python_api = paddle.tensor.matmul
        self.public_python_api = paddle.tensor.matmul
        if self.is_bfloat16_op():
            x = np.random.random(self.x_shape).astype(np.float32)
            y = np.random.random(self.y_shape).astype(np.float32)
        else:
            x = np.random.random(self.x_shape).astype(self.dtype)
            y = np.random.random(self.y_shape).astype(self.dtype)
            # -0.1 ~ 0.1
            x = -0.1 + 0.2 * x
            y = -0.1 + 0.2 * y
        result = self.reference_matmul(x, y, self.trans_x, self.trans_y)
        result = result.astype(self.dtype)
        self.inputs = {
            'X': x,
            'Y': y,
        }
        self.attrs = {'trans_x': self.trans_x, 'trans_y': self.trans_y}
        self.outputs = {'Out': result}

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(
                    place,
                    atol=0.001,
                    check_cinn=(
                        self.check_cinn if hasattr(self, 'check_cinn') else True
                    ),
                    check_pir=True,
                )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place,
                ['X', 'Y'],
                'Out',
                max_relative_error=0.1,
                check_cinn=(
                    self.check_cinn if hasattr(self, 'check_cinn') else True
                ),
                check_pir=True,
                check_prim_pir=True,
            )

    def reference_matmul(self, X, Y, transpose_X=False, transpose_Y=False):
        """Reference forward implementation using np.matmul."""
        # np.matmul does not support the transpose flags, so we manually
        # transpose X and Y appropriately.
        if transpose_X:
            if X.ndim == 1:
                X = X.reshape((X.size,))
            elif X.ndim == 2:
                X = X.T
            else:
                dim = list(range(len(X.shape)))
                dim[-1], dim[len(X.shape) - 2] = dim[len(X.shape) - 2], dim[-1]
                X = np.transpose(X, tuple(dim))
        if transpose_Y:
            if Y.ndim == 1:
                Y = Y.reshape((Y.size,))
            else:
                dim = list(range(len(Y.shape)))
                dim[-1], dim[len(Y.shape) - 2] = dim[len(Y.shape) - 2], dim[-1]
                Y = np.transpose(Y, tuple(dim))

        Out = np.matmul(X, Y)
        return Out


if __name__ == "__main__":
    unittest.main()
