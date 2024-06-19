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

import sys
import unittest

import numpy as np

sys.path.append("../../legacy_test")
from test_sparse_attention_op import get_cuda_version

from paddle.base import core

sys.path.append("..")
from op_test import OpTest, convert_float_to_uint16


class TestMulOp(OpTest):
    def setUp(self):
        self.op_type = "mul"
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'X': np.random.random((20, 5)).astype(self.dtype),
            'Y': np.random.random((5, 21)).astype(self.dtype),
        }
        self.outputs = {'Out': np.dot(self.inputs['X'], self.inputs['Y'])}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        # NODE(yjjiang11): This op will be deprecated.
        self.check_output(check_dygraph=False)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', check_dygraph=False)

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            max_relative_error=0.5,
            no_grad_set=set("X"),
            check_dygraph=False,
        )

    def test_check_grad_ignore_y(self):
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.5,
            no_grad_set=set('Y'),
            check_dygraph=False,
        )


class TestMulOp2(OpTest):
    def setUp(self):
        self.op_type = "mul"
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'X': np.random.random((3, 4, 2, 9)).astype(self.dtype),
            'Y': np.random.random((3, 6, 1, 2, 3)).astype(self.dtype),
        }
        self.attrs = {
            'x_num_col_dims': 2,
            'y_num_col_dims': 2,
        }
        result = np.dot(
            self.inputs['X'].reshape(3 * 4, 2 * 9),
            self.inputs['Y'].reshape(3 * 6, 1 * 2 * 3),
        )
        result = result.reshape(3, 4, 1, 2, 3)
        self.outputs = {'Out': result}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', check_dygraph=False)

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            max_relative_error=0.5,
            no_grad_set=set('X'),
            check_dygraph=False,
        )

    def test_check_grad_ignore_y(self):
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.5,
            no_grad_set=set('Y'),
            check_dygraph=False,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestMulFP16Op1(TestMulOp):
    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_output_with_place(place, check_dygraph=False)

    def test_check_grad_normal(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place,
                ['X', 'Y'],
                'Out',
                check_dygraph=False,
            )

    def test_check_grad_ignore_x(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place,
                ['Y'],
                'Out',
                no_grad_set=set("X"),
                check_dygraph=False,
            )

    def test_check_grad_ignore_y(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place,
                ['X'],
                'Out',
                no_grad_set=set('Y'),
                check_dygraph=False,
            )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestMulFP16Op2(TestMulOp2):
    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_output_with_place(place, check_dygraph=False)

    def test_check_grad_normal(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place,
                ['X', 'Y'],
                'Out',
                check_dygraph=False,
            )

    def test_check_grad_ignore_x(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place,
                ['Y'],
                'Out',
                no_grad_set=set("X"),
                check_dygraph=False,
            )

    def test_check_grad_ignore_y(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place,
                ['X'],
                'Out',
                no_grad_set=set('Y'),
                check_dygraph=False,
            )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestMulBF16Op1(OpTest):
    def setUp(self):
        self.op_type = "mul"
        self.init_dtype_type()
        self.inputs = {
            'X': np.random.random((20, 5)).astype(self.np_dtype),
            'Y': np.random.random((5, 21)).astype(self.np_dtype),
        }
        self.outputs = {'Out': np.dot(self.inputs['X'], self.inputs['Y'])}

        self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
        self.inputs['Y'] = convert_float_to_uint16(self.inputs['Y'])
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])
        self.place = core.CUDAPlace(0)

    def init_dtype_type(self):
        self.dtype = np.uint16
        self.np_dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place, ['X', 'Y'], 'Out', check_dygraph=False
        )

    def test_check_grad_ignore_x(self):
        self.check_grad_with_place(
            self.place,
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            check_dygraph=False,
        )

    def test_check_grad_ignore_y(self):
        self.check_grad_with_place(
            self.place,
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_dygraph=False,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestMulBF16Op2(TestMulBF16Op1):
    def setUp(self):
        self.op_type = "mul"
        self.init_dtype_type()
        self.inputs = {
            'X': np.random.random((3, 4, 2, 9)).astype(self.np_dtype),
            'Y': np.random.random((3, 6, 1, 2, 3)).astype(self.np_dtype),
        }
        self.attrs = {
            'x_num_col_dims': 2,
            'y_num_col_dims': 2,
        }
        result = np.dot(
            self.inputs['X'].reshape(3 * 4, 2 * 9),
            self.inputs['Y'].reshape(3 * 6, 1 * 2 * 3),
        )
        result = result.reshape(3, 4, 1, 2, 3)
        self.outputs = {'Out': result}

        self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
        self.inputs['Y'] = convert_float_to_uint16(self.inputs['Y'])
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])
        self.place = core.CUDAPlace(0)

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place,
            ['X', 'Y'],
            'Out',
            numeric_grad_delta=0.02,
            check_dygraph=False,
        )

    def test_check_grad_ignore_x(self):
        self.check_grad_with_place(
            self.place,
            ['Y'],
            'Out',
            numeric_grad_delta=0.02,
            no_grad_set=set("X"),
            check_dygraph=False,
        )

    def test_check_grad_ignore_y(self):
        self.check_grad_with_place(
            self.place,
            ['X'],
            'Out',
            numeric_grad_delta=0.02,
            no_grad_set=set('Y'),
            check_dygraph=False,
        )


# TODO: verify the requirements of CUDA ARCH
@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11060,
    "MatmulInt8 requires CUDA >= 11.6",
)
class TestMulInt8Op(OpTest):
    def setUp(self):
        self.op_type = "mul"
        self.dtype = np.int8
        self.init_dtype_type()
        self.inputs = {
            'X': np.random.randint(-127, 127, (8, 64)).astype(np.int32),
            'Y': np.random.randint(-127, 127, (64, 64)).astype(np.int32),
        }
        self.outputs = {'Out': np.dot(self.inputs['X'], self.inputs['Y'])}

        self.inputs['X'] = self.inputs['X'].astype(self.dtype)
        self.inputs['Y'] = self.inputs['Y'].astype(self.dtype)

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_dygraph=False)

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ignore_x(self):
        pass

    def test_check_grad_ignore_y(self):
        pass


class TestMulInt8Op2(TestMulInt8Op):
    def setUp(self):
        self.op_type = "mul"
        self.dtype = np.int8
        self.init_dtype_type()
        self.inputs = {
            'X': np.random.randint(-127, 127, (3, 4, 2, 8)).astype(np.int32),
            'Y': np.random.randint(-127, 127, (4, 4, 1, 2, 4)).astype(np.int32),
        }
        self.attrs = {
            'x_num_col_dims': 2,
            'y_num_col_dims': 2,
        }
        result = np.dot(
            self.inputs['X'].reshape(3 * 4, 2 * 8),
            self.inputs['Y'].reshape(4 * 4, 1 * 2 * 4),
        )
        result = result.reshape(3, 4, 1, 2, 4)
        self.outputs = {'Out': result}

        self.inputs['X'] = self.inputs['X'].astype(self.dtype)
        self.inputs['Y'] = self.inputs['Y'].astype(self.dtype)

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_dygraph=False)

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ignore_x(self):
        pass

    def test_check_grad_ignore_y(self):
        pass


if __name__ == "__main__":
    unittest.main()
