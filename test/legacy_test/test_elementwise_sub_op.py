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

import os
import unittest
import warnings

import numpy as np
from op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci

import paddle
from paddle import base
from paddle.base import core
from paddle.base.layer_helper import LayerHelper


class TestElementwiseOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.init_dtype()
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype),
            'Y': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}
        self.if_check_prim()
        self.if_enable_cinn()

    def init_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_prim=self.check_prim,
            check_prim_pir=self.check_prim_pir,
            check_pir=True,
        )

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            max_relative_error=0.005,
            no_grad_set=set("X"),
            check_prim=self.check_prim,
            check_prim_pir=self.check_prim_pir,
            check_pir=True,
        )

    def test_check_grad_ignore_y(self):
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.005,
            no_grad_set=set('Y'),
            check_prim=self.check_prim,
            check_prim_pir=self.check_prim_pir,
            check_pir=True,
        )

    def if_check_prim(self):
        self.check_prim = True
        self.check_prim_pir = True

    def if_enable_cinn(self):
        pass


class TestElementwiseFP16OP(TestElementwiseOp):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestElementwiseBF16OP(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.prim_op_type = "prim"
        self.dtype = np.uint16
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(np.float32),
            'Y': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(np.float32),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}
        self.inputs = {
            'X': convert_float_to_uint16(self.inputs['X']),
            'Y': convert_float_to_uint16(self.inputs['Y']),
        }
        self.outputs = {'Out': convert_float_to_uint16(self.outputs['Out'])}
        self.if_check_prim()
        self.if_enable_cinn()

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_grad_normal(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place, ['X', 'Y'], 'Out', max_relative_error=0.1
        )

    def test_check_grad_ignore_x(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            max_relative_error=0.1,
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
            max_relative_error=0.1,
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )


class TestElementwiseSubOp_ZeroDim1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.init_dtype()
        self.inputs = {
            'X': np.random.uniform(0.1, 1, []).astype(self.dtype),
            'Y': np.random.uniform(0.1, 1, []).astype(self.dtype),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}
        self.if_check_prim()
        self.if_enable_cinn()

    def if_enable_cinn(self):
        self.enable_cinn = False


class TestElementwiseSubFP16OP_ZeroDim1(TestElementwiseSubOp_ZeroDim1):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestElementwiseSubBF16OP_ZeroDim1(TestElementwiseBF16OP):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.dtype = np.uint16
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, []).astype(np.float32),
            'Y': np.random.uniform(0.1, 1, []).astype(np.float32),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}
        self.inputs = {
            'X': convert_float_to_uint16(self.inputs['X']),
            'Y': convert_float_to_uint16(self.inputs['Y']),
        }
        self.outputs = {'Out': convert_float_to_uint16(self.outputs['Out'])}
        self.if_check_prim()
        self.if_enable_cinn()


class TestElementwiseSubOp_ZeroDim2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.init_dtype()
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype),
            'Y': np.random.uniform(0.1, 1, []).astype(self.dtype),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}
        self.if_check_prim()
        self.if_enable_cinn()

    def if_enable_cinn(self):
        self.enable_cinn = False


class TestElementwiseSubFP16OP_ZeroDim2(TestElementwiseSubOp_ZeroDim2):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestElementwiseSubBF16OP_ZeroDim2(TestElementwiseBF16OP):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.dtype = np.uint16
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(np.float32),
            'Y': np.random.uniform(0.1, 1, []).astype(np.float32),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}
        self.inputs = {
            'X': convert_float_to_uint16(self.inputs['X']),
            'Y': convert_float_to_uint16(self.inputs['Y']),
        }
        self.outputs = {'Out': convert_float_to_uint16(self.outputs['Out'])}
        self.if_check_prim()
        self.if_enable_cinn()


class TestElementwiseSubOp_ZeroDim3(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.init_dtype()
        self.inputs = {
            'X': np.random.uniform(0.1, 1, []).astype(self.dtype),
            'Y': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}
        self.if_check_prim()
        self.if_enable_cinn()

    def if_enable_cinn(self):
        self.enable_cinn = False


class TestElementwiseSubFP16OP_ZeroDim3(TestElementwiseSubOp_ZeroDim3):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestElementwiseBF16OP_ZeroDim3(TestElementwiseBF16OP):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.dtype = np.uint16
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, []).astype(np.float32),
            'Y': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(np.float32),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}
        self.inputs = {
            'X': convert_float_to_uint16(self.inputs['X']),
            'Y': convert_float_to_uint16(self.inputs['Y']),
        }
        self.outputs = {'Out': convert_float_to_uint16(self.outputs['Out'])}
        self.if_check_prim()
        self.if_enable_cinn()


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestBF16ElementwiseOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.dtype = np.uint16
        x = np.random.uniform(0.1, 1, [13, 17]).astype(np.float32)
        y = np.random.uniform(0.1, 1, [13, 17]).astype(np.float32)
        out = x - y

        self.inputs = {
            'X': convert_float_to_uint16(x),
            'Y': convert_float_to_uint16(y),
        }
        self.outputs = {'Out': convert_float_to_uint16(out)}
        self.if_check_prim()
        self.if_enable_cinn()

    def if_check_prim(self):
        self.check_prim = True

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', check_prim=self.check_prim)

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Y'], 'Out', no_grad_set=set("X"), check_prim=self.check_prim
        )


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast."
)
class TestElementwiseSubOp_scalar(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.init_dtype()
        self.inputs = {
            'X': np.random.rand(10, 3, 4).astype(self.dtype),
            'Y': np.random.rand(1).astype(self.dtype),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}
        self.if_check_prim()


class TestElementwiseSubOp_Vector(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.init_dtype()
        self.inputs = {
            'X': np.random.random((100,)).astype(self.dtype),
            'Y': np.random.random((100,)).astype(self.dtype),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}
        self.if_check_prim()


class TestElementwiseSubOp_broadcast_0(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.python_api = paddle.subtract
        self.init_dtype()
        self.inputs = {
            'X': np.random.rand(100, 3, 2).astype(self.dtype),
            'Y': np.random.rand(100).astype(self.dtype),
        }

        self.attrs = {'axis': 0}
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(100, 1, 1)
        }

    def test_check_output(self):
        self.check_output(check_dygraph=False, check_pir=False)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', check_dygraph=False, check_pir=False)

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            max_relative_error=0.005,
            no_grad_set=set("X"),
            check_dygraph=False,
            check_pir=False,
        )

    def test_check_grad_ignore_y(self):
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.005,
            no_grad_set=set('Y'),
            check_dygraph=False,
            check_pir=False,
        )


class TestElementwiseSubFP16OP_broadcast_0(TestElementwiseSubOp_broadcast_0):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestElementwiseBF16OP_broadcast_0(TestElementwiseBF16OP):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.dtype = np.uint16
        self.python_api = paddle.subtract
        self.inputs = {
            'X': np.random.rand(100, 3, 2).astype(np.float32),
            'Y': np.random.rand(100).astype(np.float32),
        }
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(100, 1, 1)
        }
        self.inputs = {
            'X': convert_float_to_uint16(self.inputs['X']),
            'Y': convert_float_to_uint16(self.inputs['Y']),
        }
        self.outputs = {'Out': convert_float_to_uint16(self.outputs['Out'])}
        self.attrs = {'axis': 0}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place, check_dygraph=False, check_pir=False
        )

    def test_check_grad_normal(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place, ['X', 'Y'], 'Out', check_dygraph=False, check_pir=False
        )

    def test_check_grad_ignore_x(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            check_dygraph=False,
            check_pir=False,
        )

    def test_check_grad_ignore_y(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_dygraph=False,
            check_pir=False,
        )


class TestElementwiseSubOp_broadcast_1(TestElementwiseSubOp_broadcast_0):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.python_api = paddle.subtract
        self.init_dtype()
        self.inputs = {
            'X': np.random.rand(2, 100, 3).astype(self.dtype),
            'Y': np.random.rand(100).astype(self.dtype),
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 100, 1)
        }


class TestElementwiseSubFP16OP_broadcast_1(TestElementwiseSubOp_broadcast_1):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestElementwiseBF16OP_broadcast_1(TestElementwiseBF16OP_broadcast_0):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.dtype = np.uint16
        self.python_api = paddle.subtract
        self.inputs = {
            'X': np.random.rand(2, 100, 3).astype(np.float32),
            'Y': np.random.rand(100).astype(np.float32),
        }
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 100, 1)
        }
        self.inputs = {
            'X': convert_float_to_uint16(self.inputs['X']),
            'Y': convert_float_to_uint16(self.inputs['Y']),
        }
        self.outputs = {'Out': convert_float_to_uint16(self.outputs['Out'])}
        self.attrs = {'axis': 1}


class TestElementwiseSubOp_broadcast_2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.init_dtype()
        self.inputs = {
            'X': np.random.rand(2, 3, 100).astype(self.dtype),
            'Y': np.random.rand(100).astype(self.dtype),
        }

        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 1, 100)
        }
        self.if_check_prim()


class TestElementwiseSubFP16OP_broadcast_2(TestElementwiseSubOp_broadcast_2):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestElementwiseBF16OP_broadcast_2(TestElementwiseBF16OP_broadcast_0):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.dtype = np.uint16
        self.python_api = paddle.subtract
        self.inputs = {
            'X': np.random.rand(2, 3, 100).astype(np.float32),
            'Y': np.random.rand(100).astype(np.float32),
        }
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 1, 100)
        }
        self.inputs = {
            'X': convert_float_to_uint16(self.inputs['X']),
            'Y': convert_float_to_uint16(self.inputs['Y']),
        }
        self.outputs = {'Out': convert_float_to_uint16(self.outputs['Out'])}
        self.if_check_prim()


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestElementwiseBF16OP_broadcast_3(TestElementwiseBF16OP_broadcast_0):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.dtype = np.uint16
        self.python_api = paddle.subtract
        self.inputs = {
            'X': np.random.rand(2, 10, 12, 3).astype(np.float32),
            'Y': np.random.rand(10, 12).astype(np.float32),
        }
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 10, 12, 1)
        }
        self.inputs = {
            'X': convert_float_to_uint16(self.inputs['X']),
            'Y': convert_float_to_uint16(self.inputs['Y']),
        }
        self.outputs = {'Out': convert_float_to_uint16(self.outputs['Out'])}
        self.attrs = {'axis': 1}


class TestElementwiseSubOp_broadcast_3(TestElementwiseSubOp_broadcast_0):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.python_api = paddle.subtract
        self.init_dtype()
        self.inputs = {
            'X': np.random.rand(2, 10, 12, 3).astype(self.dtype),
            'Y': np.random.rand(10, 12).astype(self.dtype),
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 10, 12, 1)
        }


class TestElementwiseSubFP16OP_broadcast_3(TestElementwiseSubOp_broadcast_3):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseSubOp_broadcast_4(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.init_dtype()
        self.inputs = {
            'X': np.random.rand(2, 5, 3, 12).astype(self.dtype),
            'Y': np.random.rand(2, 5, 1, 12).astype(self.dtype),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}
        self.if_check_prim()
        self.if_enable_cinn()


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestElementwiseBF16OP_broadcast_4(TestElementwiseBF16OP_broadcast_0):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.dtype = np.uint16
        self.python_api = paddle.subtract
        self.inputs = {
            'X': np.random.rand(2, 5, 3, 12).astype(np.float32),
            'Y': np.random.rand(2, 5, 1, 12).astype(np.float32),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}
        self.inputs = {
            'X': convert_float_to_uint16(self.inputs['X']),
            'Y': convert_float_to_uint16(self.inputs['Y']),
        }
        self.outputs = {'Out': convert_float_to_uint16(self.outputs['Out'])}
        self.if_check_prim()


class TestElementwiseSubFP16OP_broadcast_4(TestElementwiseSubOp_broadcast_4):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseSubOp_commonuse_1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.init_dtype()
        self.inputs = {
            'X': np.random.rand(2, 3, 100).astype(self.dtype),
            'Y': np.random.rand(1, 1, 100).astype(self.dtype),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}
        self.if_check_prim()


class TestElementwiseSubFP16OP_commonuse_1(TestElementwiseSubOp_commonuse_1):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestElementwiseBF16OP_commonuse_1(TestElementwiseBF16OP):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.dtype = np.uint16
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.inputs = {
            'X': np.random.rand(2, 3, 100).astype(np.float32),
            'Y': np.random.rand(1, 1, 100).astype(np.float32),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}
        self.inputs = {
            'X': convert_float_to_uint16(self.inputs['X']),
            'Y': convert_float_to_uint16(self.inputs['Y']),
        }
        self.outputs = {'Out': convert_float_to_uint16(self.outputs['Out'])}
        self.if_check_prim()
        self.if_enable_cinn()


class TestElementwiseSubOp_commonuse_2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.init_dtype()
        self.inputs = {
            'X': np.random.rand(10, 3, 1, 4).astype(self.dtype),
            'Y': np.random.rand(10, 1, 12, 1).astype(self.dtype),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}
        self.if_check_prim()


class TestElementwiseSubFP16OP_commonuse_2(TestElementwiseSubOp_commonuse_2):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestElementwiseBF16OP_commonuse_2(TestElementwiseBF16OP):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.dtype = np.uint16
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.inputs = {
            'X': np.random.rand(10, 3, 1, 4).astype(np.float32),
            'Y': np.random.rand(10, 1, 12, 1).astype(np.float32),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}
        self.inputs = {
            'X': convert_float_to_uint16(self.inputs['X']),
            'Y': convert_float_to_uint16(self.inputs['Y']),
        }
        self.outputs = {'Out': convert_float_to_uint16(self.outputs['Out'])}
        self.if_check_prim()
        self.if_enable_cinn()


class TestElementwiseSubOp_xsize_lessthan_ysize(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.init_dtype()
        self.inputs = {
            'X': np.random.rand(10, 12).astype(self.dtype),
            'Y': np.random.rand(2, 3, 10, 12).astype(self.dtype),
        }
        self.attrs = {'axis': 2}

        self.outputs = {
            'Out': self.inputs['X'].reshape(1, 1, 10, 12) - self.inputs['Y']
        }
        self.if_check_prim()
        self.if_enable_cinn()


class TestElementwiseSubFP16OP_xsize_lessthan_ysize(
    TestElementwiseSubOp_xsize_lessthan_ysize
):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestElementwiseBF16OP_xsize_lessthan_ysize(TestElementwiseBF16OP):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.dtype = np.uint16
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.inputs = {
            'X': np.random.rand(10, 12).astype(np.float32),
            'Y': np.random.rand(2, 3, 10, 12).astype(np.float32),
        }
        self.attrs = {'axis': 2}
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}
        self.inputs = {
            'X': convert_float_to_uint16(self.inputs['X']),
            'Y': convert_float_to_uint16(self.inputs['Y']),
        }
        self.outputs = {'Out': convert_float_to_uint16(self.outputs['Out'])}
        self.if_check_prim()
        self.if_enable_cinn()


class TestComplexElementwiseSubOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"
        self.dtype = np.complex128
        self.shape = (2, 3, 4, 5)
        self.init_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {'axis': -1, 'use_mkldnn': False}
        self.outputs = {'Out': self.out}
        self.if_check_prim()
        self.if_enable_cinn()

    def init_base_dtype(self):
        self.dtype = np.complex128

    def init_input_output(self):
        self.x = np.random.random(self.shape).astype(
            self.dtype
        ) + 1j * np.random.random(self.shape).astype(self.dtype)
        self.y = np.random.random(self.shape).astype(
            self.dtype
        ) + 1j * np.random.random(self.shape).astype(self.dtype)
        self.out = self.x - self.y

    def test_check_output(self):
        self.check_output(check_pir=False)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'], 'Out', check_prim=self.check_prim, check_pir=False
        )

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            check_prim=self.check_prim,
            check_pir=False,
        )

    def test_check_grad_ignore_y(self):
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_prim=self.check_prim,
            check_pir=False,
        )

    def if_enable_cinn(self):
        self.enable_cinn = False

    def if_check_prim(self):
        self.check_prim = False


class TestRealComplexElementwiseSubOp(TestComplexElementwiseSubOp):
    def init_input_output(self):
        self.x = np.random.random(self.shape).astype(self.dtype)
        self.y = np.random.random(self.shape).astype(
            self.dtype
        ) + 1j * np.random.random(self.shape).astype(self.dtype)
        self.out = self.x - self.y

    def if_enable_cinn(self):
        self.enable_cinn = False

    def if_check_prim(self):
        self.check_prim = False


class TestSubtractApi(unittest.TestCase):
    def _executed_api(self, x, y, name=None):
        return paddle.subtract(x, y, name)

    def test_name(self):
        with paddle.pir_utils.OldIrGuard():
            with base.program_guard(base.Program()):
                x = paddle.static.data(name="x", shape=[2, 3], dtype="float32")
                y = paddle.static.data(name='y', shape=[2, 3], dtype=np.float32)

                y_1 = self._executed_api(x, y, name='subtract_res')
                self.assertEqual(('subtract_res' in y_1.name), True)

    def test_declarative(self):
        with paddle.static.program_guard(paddle.static.Program()):

            def gen_data():
                return {
                    "x": np.array([2, 3, 4]).astype(np.float32),
                    "y": np.array([1, 5, 2]).astype(np.float32),
                }

            x = paddle.static.data(name="x", shape=[3], dtype=np.float32)
            y = paddle.static.data(name="y", shape=[3], dtype=np.float32)
            z = self._executed_api(x, y)
            place = base.CPUPlace()
            exe = base.Executor(place)
            if paddle.framework.in_pir_mode():
                z_value = exe.run(feed=gen_data(), fetch_list=[z])
            else:
                z_value = exe.run(feed=gen_data(), fetch_list=[z.name])
            z_expected = np.array([1.0, -2.0, 2.0])
            self.assertEqual((z_value == z_expected).all(), True)

    def test_dygraph(self):
        with base.dygraph.guard():
            np_x = np.array([2, 3, 4]).astype('float64')
            np_y = np.array([1, 5, 2]).astype('float64')
            x = paddle.to_tensor(np_x)
            y = paddle.to_tensor(np_y)
            z = self._executed_api(x, y)
            np_z = z.numpy(False)
            z_expected = np.array([1.0, -2.0, 2.0])
            self.assertEqual((np_z == z_expected).all(), True)


class TestSubtractInplaceApi(TestSubtractApi):
    def _executed_api(self, x, y, name=None):
        return x.subtract_(y, name)


class TestSubtractInplaceBroadcastSuccess(unittest.TestCase):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 3, 4).astype('float')
        self.y_numpy = np.random.rand(3, 4).astype('float')

    def test_broadcast_success(self):
        paddle.disable_static()
        self.init_data()
        x = paddle.to_tensor(self.x_numpy)
        y = paddle.to_tensor(self.y_numpy)
        inplace_result = x.subtract_(y)
        numpy_result = self.x_numpy - self.y_numpy
        self.assertEqual((inplace_result.numpy() == numpy_result).all(), True)
        paddle.enable_static()


class TestSubtractInplaceBroadcastSuccess2(TestSubtractInplaceBroadcastSuccess):
    def init_data(self):
        self.x_numpy = np.random.rand(1, 2, 3, 1).astype('float')
        self.y_numpy = np.random.rand(3, 1).astype('float')


class TestSubtractInplaceBroadcastSuccess3(TestSubtractInplaceBroadcastSuccess):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 3, 1, 5).astype('float')
        self.y_numpy = np.random.rand(1, 3, 1, 5).astype('float')


class TestSubtractInplaceBroadcastError(unittest.TestCase):
    def init_data(self):
        self.x_numpy = np.random.rand(3, 4).astype('float')
        self.y_numpy = np.random.rand(2, 3, 4).astype('float')

    def test_broadcast_errors(self):
        paddle.disable_static()
        self.init_data()
        x = paddle.to_tensor(self.x_numpy)
        y = paddle.to_tensor(self.y_numpy)

        def broadcast_shape_error():
            x.subtract_(y)

        self.assertRaises(ValueError, broadcast_shape_error)
        paddle.enable_static()


class TestSubtractInplaceBroadcastError2(TestSubtractInplaceBroadcastError):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 1, 4).astype('float')
        self.y_numpy = np.random.rand(2, 3, 4).astype('float')


class TestSubtractInplaceBroadcastError3(TestSubtractInplaceBroadcastError):
    def init_data(self):
        self.x_numpy = np.random.rand(5, 2, 1, 4).astype('float')
        self.y_numpy = np.random.rand(2, 3, 4).astype('float')


class TestFloatElementwiseSubop(unittest.TestCase):
    def test_dygraph_sub(self):
        paddle.disable_static()

        np_a = np.random.random((2, 3, 4)).astype(np.float64)
        np_b = np.random.random((2, 3, 4)).astype(np.float64)

        tensor_a = paddle.to_tensor(np_a, dtype="float32")
        tensor_b = paddle.to_tensor(np_b, dtype="float32")

        # normal case: tensor - tensor
        expect_out = np_a - np_b
        actual_out = tensor_a - tensor_b
        np.testing.assert_allclose(
            actual_out, expect_out, rtol=1e-07, atol=1e-07
        )

        # normal case: tensor - scalar
        expect_out = np_a - 1
        actual_out = tensor_a - 1
        np.testing.assert_allclose(
            actual_out, expect_out, rtol=1e-07, atol=1e-07
        )

        # normal case: scalar - tenor
        expect_out = 1 - np_a
        actual_out = 1 - tensor_a
        np.testing.assert_allclose(
            actual_out, expect_out, rtol=1e-07, atol=1e-07
        )

        paddle.enable_static()


class TestFloatElementwiseSubop1(unittest.TestCase):
    def test_dygraph_sub(self):
        paddle.disable_static()

        np_a = np.random.random((2, 3, 4)).astype(np.float32)
        np_b = np.random.random((2, 3, 4)).astype(np.float32)

        tensor_a = paddle.to_tensor(np_a, dtype="float32")
        tensor_b = paddle.to_tensor(np_b, dtype="float32")

        # normal case: nparray - tenor
        expect_out = np_a - np_b
        actual_out = np_a - tensor_b
        np.testing.assert_allclose(
            actual_out, expect_out, rtol=1e-07, atol=1e-07
        )

        # normal case: tenor - nparray
        actual_out = tensor_a - np_b
        np.testing.assert_allclose(
            actual_out, expect_out, rtol=1e-07, atol=1e-07
        )

        paddle.enable_static()


class TestTensorSubAPIWarnings(unittest.TestCase):
    def test_warnings(self):
        with paddle.pir_utils.OldIrGuard():
            with warnings.catch_warnings(record=True) as context:
                warnings.simplefilter("always")

                paddle.enable_static()
                helper = LayerHelper("elementwise_sub")
                data = paddle.static.data(
                    name='data', shape=[None, 3, 32, 32], dtype=np.float32
                )
                out = helper.create_variable_for_type_inference(
                    dtype=data.dtype
                )
                os.environ['FLAGS_print_extra_attrs'] = "1"
                helper.append_op(
                    type="elementwise_sub",
                    inputs={'X': data, 'Y': data},
                    outputs={'Out': out},
                    attrs={'axis': 1, 'use_mkldnn': False},
                )
                self.assertTrue(
                    "op elementwise_sub's attr axis = 1 is not the default value: -1"
                    in str(context[-1].message)
                )
                os.environ['FLAGS_print_extra_attrs'] = "0"


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
