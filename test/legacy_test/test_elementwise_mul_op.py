#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci

import paddle
from paddle.base import core


class ElementwiseMulOp(OpTest):
    def init_kernel_type(self):
        self.use_mkldnn = False

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

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_output(
            check_dygraph=(not self.use_mkldnn),
            check_pir=(not self.use_mkldnn),
        )

    def test_check_grad_normal(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_dygraph=(not self.use_mkldnn),
            check_prim=True,
            check_prim_pir=(not self.use_mkldnn),
            check_pir=(not self.use_mkldnn),
        )

    def test_check_grad_ingore_x(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            check_dygraph=(not self.use_mkldnn),
            check_prim=True,
            check_prim_pir=(not self.use_mkldnn),
            check_pir=(not self.use_mkldnn),
        )

    def test_check_grad_ingore_y(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_dygraph=(not self.use_mkldnn),
            check_prim=True,
            check_prim_pir=(not self.use_mkldnn),
            check_pir=(not self.use_mkldnn),
        )

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    def init_dtype(self):
        pass

    def init_axis(self):
        pass

    def if_enable_cinn(self):
        pass


class TestComplexElementwiseMulOpWithCheckGrad(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.prim_op_type = "prim"
        self.python_api = paddle.multiply
        self.public_python_api = paddle.multiply
        self.dtype = np.complex128
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
        self.attrs = {'axis': self.axis}

    def init_input_output(self):
        self.x = np.array([3 + 4j, 1 + 2j]).astype(self.dtype)
        self.y = np.array([3 + 4j, 5 + 6j]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', check_pir=True)

    def test_check_grad_ingore_x(self):
        self.check_grad(['Y'], 'Out', no_grad_set=set("X"), check_pir=True)

    def test_check_grad_ingore_y(self):
        self.check_grad(['X'], 'Out', no_grad_set=set('Y'), check_pir=True)


class TestElementwiseMulOp_ZeroDim1(ElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, []).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, []).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)


class TestElementwiseMulOp_ZeroDim2(ElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, []).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)


class TestElementwiseMulOp_ZeroDim3(ElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, []).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or paddle.is_compiled_with_rocm(),
    "BFP16 test runs only on CUDA",
)
class TestBF16ElementwiseMulOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.prim_op_type = "prim"
        self.python_api = paddle.multiply
        self.public_python_api = paddle.multiply
        self.dtype = np.uint16

        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(np.float32)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(np.float32)
        self.out = np.multiply(self.x, self.y)

        self.axis = -1

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(convert_float_to_uint16(self.x)),
            'Y': OpTest.np_dtype_to_base_dtype(convert_float_to_uint16(self.y)),
        }
        self.outputs = {'Out': convert_float_to_uint16(self.out)}
        self.attrs = {'axis': self.axis, 'use_mkldnn': False}
        self.if_enable_cinn()

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )

    def if_enable_cinn(self):
        self.enable_cinn = False


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast."
)
class TestElementwiseMulOp_scalar(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.prim_op_type = "prim"
        self.python_api = paddle.multiply
        self.public_python_api = paddle.multiply
        self.inputs = {
            'X': np.random.rand(10, 3, 4).astype(np.float64),
            'Y': np.random.rand(1).astype(np.float64),
        }
        self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}
        self.init_kernel_type()


class TestElementwiseMulOp_Vector(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.prim_op_type = "prim"
        self.python_api = paddle.multiply
        self.public_python_api = paddle.multiply
        self.inputs = {
            'X': np.random.random((100,)).astype("float64"),
            'Y': np.random.random((100,)).astype("float64"),
        }
        self.outputs = {'Out': np.multiply(self.inputs['X'], self.inputs['Y'])}
        self.init_kernel_type()


class ElementwiseMulOp_broadcast(OpTest):
    def init_kernel_type(self):
        self.use_mkldnn = False

    def setUp(self):
        self.op_type = "elementwise_mul"
        self.prim_op_type = "prim"
        self.python_api = paddle.multiply
        self.public_python_api = paddle.multiply
        self.init_dtype()
        self.init_kernel_type()
        self.init_axis()
        self.init_input_attr_output()
        self.if_check_prim()
        self.if_check_dygraph()

    def test_check_output(self):
        self.check_output(
            check_dygraph=self.check_dygraph,
            check_pir=self.check_dygraph,
        )

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_dygraph=self.check_dygraph,
            check_prim=self.check_prim,
            check_pir=self.check_dygraph,
        )

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            check_dygraph=self.check_dygraph,
            check_prim=self.check_prim,
            check_pir=self.check_dygraph,
        )

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_dygraph=self.check_dygraph,
            check_prim=self.check_prim,
            check_pir=self.check_dygraph,
        )

    def init_input_attr_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17, 1]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [17, 17]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)
        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.outputs = {'Out': self.out}
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}

    def init_dtype(self):
        self.dtype = np.float64

    def init_axis(self):
        self.axis = -1

    def if_check_prim(self):
        self.check_prim = self.axis == -1

    def if_check_dygraph(self):
        self.check_dygraph = (not self.use_mkldnn) and (self.axis == -1)


class TestElementwiseMulOp_broadcast_0(ElementwiseMulOp_broadcast):
    def init_input_attr_output(self):
        self.x = np.random.rand(100, 2, 3).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x * self.y.reshape(100, 1, 1)
        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.outputs = {'Out': self.out}
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}

    def init_axis(self):
        self.axis = 0


class TestElementwiseMulOp_broadcast_1(ElementwiseMulOp_broadcast):
    def init_input_attr_output(self):
        self.inputs = {
            'X': np.random.rand(2, 100, 3).astype(np.float64),
            'Y': np.random.rand(100).astype(np.float64),
        }

        self.attrs = {'axis': self.axis}
        self.outputs = {
            'Out': self.inputs['X'] * self.inputs['Y'].reshape(1, 100, 1)
        }

    def init_axis(self):
        self.axis = 1


class TestElementwiseMulOp_broadcast_2(ElementwiseMulOp_broadcast):
    def init_input_attr_output(self):
        self.inputs = {
            'X': np.random.rand(2, 3, 100).astype(np.float64),
            'Y': np.random.rand(100).astype(np.float64),
        }
        self.attrs = {'axis': self.axis}
        self.outputs = {
            'Out': self.inputs['X'] * self.inputs['Y'].reshape(1, 1, 100)
        }


class TestElementwiseMulOp_broadcast_3(ElementwiseMulOp_broadcast):
    def init_input_attr_output(self):
        self.inputs = {
            'X': np.random.rand(2, 10, 12, 3).astype(np.float64),
            'Y': np.random.rand(10, 12).astype(np.float64),
        }

        self.attrs = {'axis': self.axis}
        self.outputs = {
            'Out': self.inputs['X'] * self.inputs['Y'].reshape(1, 10, 12, 1)
        }

    def init_axis(self):
        self.axis = 1


class TestElementwiseMulOp_broadcast_4(ElementwiseMulOp_broadcast):
    def init_input_attr_output(self):
        self.inputs = {
            'X': np.random.rand(10, 2, 11).astype(np.float64),
            'Y': np.random.rand(10, 1, 11).astype(np.float64),
        }
        self.attrs = {'axis': self.axis}
        self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}


class TestElementwiseMulOp_broadcast_5(ElementwiseMulOp_broadcast):
    def init_input_attr_output(self):
        self.inputs = {
            'X': np.random.rand(10, 4, 2, 3).astype(np.float64),
            'Y': np.random.rand(10, 4, 1, 3).astype(np.float64),
        }
        self.attrs = {'axis': self.axis}
        self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestElementwiseMulOpFp16(ElementwiseMulOp):
    def init_dtype(self):
        self.dtype = np.float16

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_output(check_dygraph=(not self.use_mkldnn))

    def test_check_grad_normal(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_dygraph=(not self.use_mkldnn),
            check_prim=True,
            check_prim_pir=(not self.use_mkldnn),
            check_pir=(not self.use_mkldnn),
        )

    def test_check_grad_ingore_x(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            check_dygraph=(not self.use_mkldnn),
            check_prim=True,
            check_prim_pir=(not self.use_mkldnn),
            check_pir=(not self.use_mkldnn),
        )

    def test_check_grad_ingore_y(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_dygraph=(not self.use_mkldnn),
            check_prim=True,
            check_prim_pir=(not self.use_mkldnn),
            check_pir=(not self.use_mkldnn),
        )


class TestElementwiseMulOp_commonuse_1(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.prim_op_type = "prim"
        self.python_api = paddle.multiply
        self.public_python_api = paddle.multiply
        self.inputs = {
            'X': np.random.rand(2, 3, 100).astype(np.float64),
            'Y': np.random.rand(1, 1, 100).astype(np.float64),
        }
        self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}
        self.init_kernel_type()


class TestElementwiseMulOp_commonuse_2(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.prim_op_type = "prim"
        self.python_api = paddle.multiply
        self.public_python_api = paddle.multiply
        self.inputs = {
            'X': np.random.rand(30, 3, 1, 5).astype(np.float64),
            'Y': np.random.rand(30, 1, 4, 1).astype(np.float64),
        }
        self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}
        self.init_kernel_type()


class TestElementwiseMulOp_xsize_lessthan_ysize(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.prim_op_type = "prim"
        self.python_api = paddle.multiply
        self.public_python_api = paddle.multiply
        self.inputs = {
            'X': np.random.rand(10, 10).astype(np.float64),
            'Y': np.random.rand(2, 2, 10, 10).astype(np.float64),
        }

        self.attrs = {'axis': 2}

        self.outputs = {
            'Out': self.inputs['X'].reshape(1, 1, 10, 10) * self.inputs['Y']
        }
        self.init_kernel_type()


class TestComplexElementwiseMulOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.prim_op_type = "prim"
        self.python_api = paddle.multiply
        self.init_base_dtype()
        self.init_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {'axis': -1, 'use_mkldnn': False}
        self.outputs = {'Out': self.out}

    def init_base_dtype(self):
        self.dtype = np.complex128

    def init_input_output(self):
        self.x = np.random.random((2, 3, 4, 5)).astype(
            self.dtype
        ) + 1j * np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.y = np.random.random((2, 3, 4, 5)).astype(
            self.dtype
        ) + 1j * np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.out = self.x * self.y

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', check_pir=True)

    def test_check_grad_ingore_x(self):
        self.check_grad(['Y'], 'Out', no_grad_set=set("X"), check_pir=True)

    def test_check_grad_ingore_y(self):
        self.check_grad(['X'], 'Out', no_grad_set=set('Y'), check_pir=True)


class TestRealComplexElementwiseMulOp(TestComplexElementwiseMulOp):
    def init_base_dtype(self):
        self.dtype = np.complex128

    def init_input_output(self):
        self.x = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.y = np.random.random((2, 3, 4, 5)).astype(
            self.dtype
        ) + 1j * np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.out = self.x * self.y


class TestElementwiseMulop(unittest.TestCase):
    def test_dygraph_mul(self):
        paddle.disable_static()

        np_a = np.random.random((2, 3, 4)).astype(np.float32)
        np_b = np.random.random((2, 3, 4)).astype(np.float32)

        tensor_a = paddle.to_tensor(np_a, dtype="float32")
        tensor_b = paddle.to_tensor(np_b, dtype="float32")

        # normal case: nparray * tenor
        expect_out = np_a * np_b
        actual_out = np_a * tensor_b
        np.testing.assert_allclose(actual_out, expect_out)

        # normal case: tensor * nparray
        actual_out = tensor_a * np_b
        np.testing.assert_allclose(actual_out, expect_out)

        paddle.enable_static()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
