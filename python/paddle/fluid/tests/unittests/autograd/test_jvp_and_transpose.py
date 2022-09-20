# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers.utils import flatten
from paddle.incubate.autograd.primrules import _orig2prim, _prim2orig, _jvp, _transpose

paddle.enable_static()


############################ Test linearize rules ############################
class TestAddPJVPAndTranspose(unittest.TestCase):

    def setUp(self):
        self.main_program = paddle.static.Program()
        self.startup_program = paddle.static.Program()
        self.layer_help = LayerHelper('TestPrim2Orig')

        with paddle.static.program_guard(self.main_program,
                                         self.startup_program):
            self.init_data()

    def init_data(self):
        # Set prim op
        self.op_type = 'add_p'
        X = paddle.static.data(name='X', shape=[2, 2], dtype='float')
        Y = paddle.static.data(name='Y', shape=[2, 2], dtype='float')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {
            'Z':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[2, 2], dtype='float')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[2, 2], dtype='float')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}

        # Set transpose
        check_dot = lambda v: True
        Z_BAR = paddle.static.data(name='Z_BAR', shape=[2, 2], dtype='float')
        self.transpose_args = (check_dot, Z_BAR)
        self.transpose_out_shape_map = {0: X, 1: Y}

        self.all_ops = [
            # prim op:
            'add_p',
            # jvp op:
            'add_p',
            # transpose op:
        ]

    def test_op(self):
        with paddle.static.program_guard(self.main_program,
                                         self.startup_program):
            op = self.layer_help.append_op(type=self.op_type,
                                           inputs=self.prim_input,
                                           outputs=self.prim_output,
                                           attrs=self.prim_attrs)

            jvp_out = _jvp(op, *self.jvp_args)
            jvp_out = flatten(jvp_out)
            for k, v in self.jvp_out_shape_map.items():
                self.assertEqual(jvp_out[k].shape, v.shape)

            # Some prim ops dont have transpose rule
            if hasattr(self, 'transpose_args'):
                transpose_out = _transpose(op, *self.transpose_args)
                transpose_out = flatten(transpose_out)
                for k, v in self.transpose_out_shape_map.items():
                    self.assertEqual(transpose_out[k].shape, v.shape)

            all_ops = [op.type for op in self.main_program.block(0).ops]
            self.assertEqual(sorted(all_ops), sorted(self.all_ops))


class TestSubPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'sub_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        Y = paddle.static.data(name='Y', shape=[5, 6], dtype='int64')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {
            'Z':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}

        # Set transpose
        check_dot = lambda v: True
        Z_BAR = paddle.static.data(name='Z_BAR', shape=[5, 6], dtype='int64')
        self.transpose_args = (check_dot, Z_BAR)
        self.transpose_out_shape_map = {0: X, 1: Y}

        self.all_ops = [
            # prim op:
            'sub_p',
            # jvp op:
            'sub_p',
            # transpose op:
            'fill_constant_p',
            'sub_p'
        ]


class TestMulPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'mul_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        Y = paddle.static.data(name='Y', shape=[5, 6], dtype='int64')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {
            'Z':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}

        # Set transpose
        check_dot = lambda v: v is X
        Z_BAR = paddle.static.data(name='Z_BAR', shape=[5, 6], dtype='int64')
        self.transpose_args = (check_dot, Z_BAR)
        self.transpose_out_shape_map = {
            0: X,
        }

        self.all_ops = [
            # prim op:
            'mul_p',
            # jvp op:
            'mul_p',
            'mul_p',
            'add_p',
            # transpose op:
            'mul_p'
        ]


class TestDivPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'div_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        Y = paddle.static.data(name='Y', shape=[5, 6], dtype='int64')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {
            'Z':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}

        # Set transpose
        check_dot = lambda v: v is X
        Z_BAR = paddle.static.data(name='Z_BAR', shape=[5, 6], dtype='int64')
        self.transpose_args = (check_dot, Z_BAR)
        self.transpose_out_shape_map = {
            0: X,
        }

        self.all_ops = [
            # prim op:
            'div_p',
            # jvp op:
            'div_p',
            'div_p',
            'mul_p',
            'mul_p',
            'sub_p',
            # transpose op:
            'div_p'
        ]


class TestSqrtPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'sqrt_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {
            'X': X,
        }
        self.prim_output = {
            'Y':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT, )
        self.jvp_out_shape_map = {0: self.prim_output['Y']}

        self.all_ops = [
            # prim op:
            'sqrt_p',
            # jvp op:
            'div_p',
            'mul_p',
            'fill_constant_p',
            # 'sqrt_p',
            # transpose op:
        ]


class TestTanhPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'tanh_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {
            'X': X,
        }
        self.prim_output = {
            'Y':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT, )
        self.jvp_out_shape_map = {0: self.prim_output['Y']}

        self.all_ops = [
            # prim op:
            'tanh_p',
            # jvp op:
            'mul_p',
            'sub_p',
            'fill_constant_p',
            'mul_p',
            # transpose op:
        ]


class TestSinPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'sin_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {
            'X': X,
        }
        self.prim_output = {
            'Y':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT, )
        self.jvp_out_shape_map = {0: self.prim_output['Y']}

        self.all_ops = [
            # prim op:
            'sin_p',
            # jvp op:
            'mul_p',
            'cos_p',
            # transpose op:
        ]


class TestCosPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'cos_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {
            'X': X,
        }
        self.prim_output = {
            'Y':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT, )
        self.jvp_out_shape_map = {0: self.prim_output['Y']}

        self.all_ops = [
            # prim op:
            'cos_p',
            # jvp op:
            'mul_p',
            'sin_p',
            'fill_constant_p',
            'sub_p'
            # transpose op:
        ]


class TestExpPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'exp_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {
            'X': X,
        }
        self.prim_output = {
            'Y':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT, )
        self.jvp_out_shape_map = {0: self.prim_output['Y']}

        self.all_ops = [
            # prim op:
            'exp_p',
            # jvp op:
            'mul_p',
            # transpose op:
        ]


class TestErfPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'erf_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {
            'X': X,
        }
        self.prim_output = {
            'Y':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT, )
        self.jvp_out_shape_map = {0: self.prim_output['Y']}

        self.all_ops = [
            # prim op:
            'erf_p',
            # jvp op:
            'exp_p',
            'fill_constant_p',
            'fill_constant_p',
            'fill_constant_p',
            'mul_p',
            'mul_p',
            'pow_p',
            'sub_p',
            # transpose op:
        ]


class TestAbsPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'abs_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {
            'X': X,
        }
        self.prim_output = {
            'Y':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT, )
        self.jvp_out_shape_map = {0: self.prim_output['Y']}

        self.all_ops = [
            # prim op:
            'abs_p',
            # jvp op:
            'select_p',
            'ge_p',
            'fill_constant_p',
            'fill_constant_p',
            'sub_p',
            # transpose op:
        ]


class TestCastPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'cast_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {
            'X': X,
        }
        self.prim_output = {
            'Y':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {'dtype': paddle.float64}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT, )
        self.jvp_out_shape_map = {0: self.prim_output['Y']}

        # Set transpose
        check_dot = lambda v: True
        Y_BAR = paddle.static.data(name='Y_BAR', shape=[5, 6], dtype='float')
        self.transpose_args = (check_dot, Y_BAR)
        self.transpose_out_shape_map = {0: X}

        self.all_ops = [
            # prim op:
            'cast_p',
            # jvp op:
            'cast_p',
            # transpose op:
            'cast_p'
        ]


class TestLogPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'log_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {
            'X': X,
        }
        self.prim_output = {
            'Y':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT, )
        self.jvp_out_shape_map = {0: self.prim_output['Y']}

        self.all_ops = [
            # prim op:
            'log_p',
            # jvp op:
            'div_p',
            # transpose op:
        ]


class TestReshapePJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'reshape_p'
        X = paddle.static.data(name='X', shape=[8, 8], dtype='int64')
        self.prim_input = {
            'X': X,
        }
        self.prim_output = {
            'Y':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {'shape': [2, 32]}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[8, 8], dtype='int64')
        self.jvp_args = (X_DOT, )
        self.jvp_out_shape_map = {0: self.prim_output['Y']}

        # Set transpose
        check_dot = lambda v: v is X
        Y_BAR = paddle.static.data(name='Y_BAR', shape=[2, 32], dtype='int64')
        self.transpose_args = (check_dot, Y_BAR)
        self.transpose_out_shape_map = {
            0: X,
        }

        self.all_ops = [
            # prim op:
            'reshape_p',
            # jvp op:
            'reshape_p',
            # transpose op:
            'reshape_p',
        ]


class TestBroadcastPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'broadcast_p'
        X = paddle.static.data(name='X', shape=[10, 1], dtype='int64')
        self.prim_input = {
            'X': X,
        }
        self.prim_output = {
            'Y':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {'shape': [2, 10, 7]}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[10, 7], dtype='int64')
        self.jvp_args = (X_DOT, )
        self.jvp_out_shape_map = {0: self.prim_output['Y']}

        # Set transpose
        check_dot = lambda v: v is X
        Y_BAR = paddle.static.data(name='Y_BAR',
                                   shape=[2, 10, 7],
                                   dtype='int64')
        self.transpose_args = (check_dot, Y_BAR)
        self.transpose_out_shape_map = {
            0: X,
        }

        self.all_ops = [
            # prim op:
            'broadcast_p',
            # jvp op:
            'broadcast_p',
            # transpose op:
            'reduce_sum_p',
            'reshape_p'
        ]


class TestTransposePJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'transpose_p'
        X = paddle.static.data(name='X', shape=[2, 3, 4, 5], dtype='int64')
        self.prim_input = {
            'X': X,
        }
        self.prim_output = {
            'Y':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {'axis': [0, 2, 3, 1]}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT',
                                   shape=[2, 3, 4, 5],
                                   dtype='int64')
        self.jvp_args = (X_DOT, )
        self.jvp_out_shape_map = {0: self.prim_output['Y']}

        # Set transpose
        check_dot = lambda v: v is X
        Y_BAR = paddle.static.data(name='Y_BAR',
                                   shape=[2, 4, 5, 3],
                                   dtype='int64')
        self.transpose_args = (check_dot, Y_BAR)
        self.transpose_out_shape_map = {
            0: X,
        }

        self.all_ops = [
            # prim op:
            'transpose_p',
            # jvp op:
            'transpose_p',
            # transpose op:
            'transpose_p',
        ]


class TestSplitPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'split_p'
        X = paddle.static.data(name='X', shape=[2, 7, 10], dtype='int64')
        self.prim_input = {
            'X': X,
        }
        self.prim_output = {
            'YS': [
                self.layer_help.create_variable_for_type_inference(
                    dtype=X.dtype) for i in range(4)
            ]
        }
        self.prim_attrs = {'num_or_sections': [2, 3, 4, 1], 'axis': 2}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT',
                                   shape=[2, 7, 10],
                                   dtype='int64')
        self.jvp_args = (X_DOT, )
        self.jvp_out_shape_map = {
            0: self.prim_output['YS'][0],
            1: self.prim_output['YS'][1],
            2: self.prim_output['YS'][2],
            3: self.prim_output['YS'][3],
        }

        # Set transpose
        check_dot = lambda v: v is X
        YS_BAR = [
            paddle.static.data(name='Y_BAR1', shape=[2, 7, 2], dtype='int64'),
            paddle.static.data(name='Y_BAR2', shape=[2, 7, 3], dtype='int64'),
            paddle.static.data(name='Y_BAR3', shape=[2, 7, 4], dtype='int64'),
            paddle.static.data(name='Y_BAR4', shape=[2, 7, 1], dtype='int64'),
        ]
        self.transpose_args = (check_dot, YS_BAR)
        self.transpose_out_shape_map = {
            0: X,
        }

        self.all_ops = [
            # prim op:
            'split_p',
            # jvp op:
            'split_p',
            # transpose op:
            'concat_p',
        ]


class TestConcatPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'concat_p'
        X = paddle.static.data(name='X', shape=[3, 9, 5], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[3, 2, 5], dtype='float64')
        Z = paddle.static.data(name='Z', shape=[3, 3, 5], dtype='float64')
        self.prim_input = {
            'XS': [X, Y, Z],
        }
        self.prim_output = {
            'Y':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {'axis': 1}

        # Set JVP
        XS_DOT = [
            paddle.static.data(name='X_DOT1', shape=[3, 9, 5], dtype='float64'),
            paddle.static.data(name='X_DOT2', shape=[3, 2, 5], dtype='float64'),
            paddle.static.data(name='X_DOT3', shape=[3, 3, 5], dtype='float64'),
        ]
        self.jvp_args = (XS_DOT, )
        self.jvp_out_shape_map = {0: self.prim_output['Y']}

        # Set transpose
        check_dot = lambda v: v is X or v is Y or v is Z
        Y_BAR = paddle.static.data(name='Y_BAR',
                                   shape=[3, 14, 5],
                                   dtype='float64')
        self.transpose_args = (check_dot, Y_BAR)
        self.transpose_out_shape_map = {
            0: X,
            1: Y,
            2: Z,
        }

        self.all_ops = [
            # prim op:
            'concat_p',
            # jvp op:
            'concat_p',
            # transpose op:
            'split_p',
        ]


class TestReduceSumPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'reduce_sum_p'
        X = paddle.static.data(name='X', shape=[2, 3, 4, 5], dtype='float64')
        self.prim_input = {'X': X}
        self.prim_output = {
            'Y':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {'axis': [2], 'keepdim': False}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT1',
                                   shape=[2, 3, 4, 5],
                                   dtype='float64')
        self.jvp_args = (X_DOT, )
        self.jvp_out_shape_map = {0: self.prim_output['Y']}

        # Set transpose
        check_dot = lambda v: v is X
        Y_BAR = paddle.static.data(name='Y_BAR',
                                   shape=[2, 3, 5],
                                   dtype='float64')
        self.transpose_args = (check_dot, Y_BAR)
        self.transpose_out_shape_map = {
            0: X,
        }

        self.all_ops = [
            # prim op:
            'reduce_sum_p',
            # jvp op:
            'reduce_sum_p',
            # transpose op:
            'reshape_p',
            'broadcast_p',
        ]


class TestMatmulPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'matmul_p'
        X = paddle.static.data(name='X', shape=[2, 3], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[3, 4], dtype='float64')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {
            'Z':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[2, 3], dtype='float64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[3, 4], dtype='float64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}

        # Set transpose
        check_dot = lambda v: v is X
        Z_BAR = paddle.static.data(name='Z_BAR', shape=[2, 4], dtype='float64')
        self.transpose_args = (check_dot, Z_BAR)
        self.transpose_out_shape_map = {
            0: X,
        }

        self.all_ops = [
            # prim op:
            'matmul_p',
            # jvp op:
            'matmul_p',
            'matmul_p',
            'add_p',
            # transpose op:
            'matmul_p',
            'transpose_p',
        ]


class TestSliceSelectPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'slice_select_p'
        X = paddle.static.data(name='X', shape=[3, 20], dtype='float64')
        self.prim_input = {
            'X': X,
        }
        self.prim_output = {
            'Y':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {
            'axis': [1],
            'starts': [0],
            'ends': [20],
            'strides': [2]
        }

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[3, 20], dtype='float64')
        self.jvp_args = (X_DOT, )
        self.jvp_out_shape_map = {0: self.prim_output['Y']}

        # Set transpose
        check_dot = lambda v: v is X
        Y_BAR = paddle.static.data(name='Y_BAR', shape=[3, 10], dtype='float64')
        self.transpose_args = (check_dot, Y_BAR)
        self.transpose_out_shape_map = {
            0: X,
        }

        self.all_ops = [
            # prim op:
            'slice_select_p',
            # jvp op:
            'slice_select_p',
            # transpose op:
            'slice_assign_p',
            'fill_constant_p',
        ]


class TestSliceAssignPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'slice_assign_p'
        X = paddle.static.data(name='X', shape=[3, 20], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[3, 5], dtype='float64')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {
            'Z':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {
            'axis': [1],
            'starts': [0],
            'ends': [10],
            'strides': [2]
        }

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[3, 20], dtype='float64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[3, 5], dtype='float64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}

        # Set transpose
        check_dot = lambda v: v is X or v is Y
        Z_BAR = paddle.static.data(name='Z_BAR', shape=[3, 20], dtype='float64')
        self.transpose_args = (check_dot, Z_BAR)
        self.transpose_out_shape_map = {0: X, 1: Y}

        self.all_ops = [
            # prim op:
            'slice_assign_p',
            # jvp op:
            'slice_assign_p',
            # transpose op:
            'slice_assign_p',
            'slice_select_p',
            'fill_constant_p'
        ]


class TestGatherPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'gather_p'
        X = paddle.static.data(name='X', shape=[9, 5], dtype='float64')
        IndexTensor = paddle.static.data(name='IndexTensor',
                                         shape=[3],
                                         dtype='int32')
        self.prim_input = {'X': X, 'IndexTensor': IndexTensor}
        self.prim_output = {
            'Y':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {'axis': 1}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[9, 5], dtype='float64')
        self.jvp_args = (
            X_DOT,
            IndexTensor,
        )
        self.jvp_out_shape_map = {0: self.prim_output['Y']}

        # Set transpose
        check_dot = lambda v: v is X
        Y_BAR = paddle.static.data(name='Y_BAR', shape=[9, 3], dtype='float64')
        self.transpose_args = (check_dot, Y_BAR)
        self.transpose_out_shape_map = {
            0: X,
        }

        self.all_ops = [
            # prim op:
            'gather_p',
            # jvp op:
            'gather_p',
            # transpose op:
            'scatter_add_p',
            'fill_constant_p',
        ]


class TestScatterAddPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'scatter_add_p'
        X = paddle.static.data(name='X', shape=[9, 5], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[9, 3], dtype='float64')
        IndexTensor = paddle.static.data(name='IndexTensor',
                                         shape=[3],
                                         dtype='int32')
        self.prim_input = {'X': X, 'Y': Y, 'IndexTensor': IndexTensor}
        self.prim_output = {
            'Z':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {'axis': 1}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[9, 5], dtype='float64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[9, 3], dtype='float64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}

        # Set transpose
        check_dot = lambda v: v is X or v is Y
        Z_BAR = paddle.static.data(name='Z_BAR', shape=[9, 5], dtype='float64')
        self.transpose_args = (check_dot, Z_BAR)
        self.transpose_out_shape_map = {0: X, 1: Y}

        self.all_ops = [
            # prim op:
            'scatter_add_p',
            # jvp op:
            'scatter_add_p',
            # transpose op:
            'scatter_add_p',
            'fill_constant_p',
            'gather_p'
        ]


class TestSelectPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'select_p'
        Cond = paddle.static.data(name='Condition', shape=[9, 5], dtype='bool')
        X = paddle.static.data(name='X', shape=[9, 5], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[9, 5], dtype='float64')

        self.prim_input = {'Condition': Cond, 'X': X, 'Y': Y}
        self.prim_output = {
            'Z':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        Cond_DOT = paddle.static.data(name='Cond_DOT',
                                      shape=[9, 5],
                                      dtype='float64')
        X_DOT = paddle.static.data(name='X_DOT', shape=[9, 5], dtype='float64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[9, 5], dtype='float64')
        self.jvp_args = (Cond_DOT, X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}

        # Set transpose
        check_dot = lambda v: True
        Z_BAR = paddle.static.data(name='Z_BAR', shape=[9, 5], dtype='float64')
        self.transpose_args = (check_dot, Z_BAR)
        self.transpose_out_shape_map = {0: X, 1: Y}

        self.all_ops = [
            # prim op:
            'select_p',
            # jvp op:
            'select_p',
            # transpose op:
            'fill_constant_p',
            'fill_constant_p',
            'fill_constant_p',
            'select_p',
            'select_p',
        ]


class TestEqPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'eq_p'
        X = paddle.static.data(name='X', shape=[4, 5], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[4, 5], dtype='float64')

        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {
            'Z':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[4, 5], dtype='float64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[4, 5], dtype='float64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}

        self.all_ops = [
            # prim op:
            'eq_p',
            # jvp op:
            'fill_constant_p',
            # transpose op:
        ]


class TestGtPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'gt_p'
        X = paddle.static.data(name='X', shape=[4, 5], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[4, 5], dtype='float64')

        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {
            'Z':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[4, 5], dtype='float64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[4, 5], dtype='float64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}

        self.all_ops = [
            # prim op:
            'gt_p',
            # jvp op:
            'fill_constant_p',
            # transpose op:
        ]


class TestGePJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'ge_p'
        X = paddle.static.data(name='X', shape=[4, 5], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[4, 5], dtype='float64')

        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {
            'Z':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[4, 5], dtype='float64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[4, 5], dtype='float64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}

        self.all_ops = [
            # prim op:
            'ge_p',
            # jvp op:
            'fill_constant_p',
            # transpose op:
        ]


class TestNePJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'ne_p'
        X = paddle.static.data(name='X', shape=[4, 5], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[4, 5], dtype='float64')

        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {
            'Z':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[4, 5], dtype='float64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[4, 5], dtype='float64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}

        self.all_ops = [
            # prim op:
            'ne_p',
            # jvp op:
            'fill_constant_p',
            # transpose op:
        ]


class TestPowPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'pow_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='float32')
        Y = paddle.static.data(name='Y', shape=[5, 6], dtype='float32')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {
            'Z':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='float32')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[5, 6], dtype='float32')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}

        self.all_ops = [
            # prim op:
            'pow_p',
            # jvp op:
            'fill_constant_p',
            'fill_constant_p',
            'eq_p',
            'select_p',
            'sub_p',
            'mul_p',
            'mul_p',
            'pow_p',
            'mul_p',
            'mul_p',
            'log_p',
            'add_p'
            # transpose op:
        ]


class TestMaxPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        # Set prim op
        self.op_type = 'max_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='float32')
        Y = paddle.static.data(name='Y', shape=[5, 6], dtype='float32')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {
            'Z':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.prim_attrs = {}

        # Set JVP
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='float32')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[5, 6], dtype='float32')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}

        self.all_ops = [
            # prim op:
            'max_p',
            # jvp op:
            'fill_constant_p',
            'eq_p',
            'select_p',
            # transpose op:
        ]


if __name__ == '__main__':
    unittest.main()
