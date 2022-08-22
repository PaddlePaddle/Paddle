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


############################ Test orig2prim rules ############################
class TestElementWiseAddOrig2Prim(unittest.TestCase):

    def setUp(self):
        self.main_program = paddle.static.Program()
        self.startup_program = paddle.static.Program()
        self.layer_help = LayerHelper('TestOrig2Prim')

        with paddle.static.program_guard(self.main_program,
                                         self.startup_program):
            self.init_data()

    def init_data(self):
        self.op_type = 'elementwise_add'
        X = paddle.static.data(name='X', shape=[2, 2], dtype='float')
        Y = paddle.static.data(name='Y', shape=[2, 2], dtype='float')

        self.input = {'X': X, 'Y': Y}
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {}

        self.orig2prim_args = (X, Y)
        self.all_ops = ['elementwise_add', 'add_p']
        # { prim_op_output_index: orig_op_output_var }
        self.out_map = {0: self.output['Out']}

    def test_op(self):
        with paddle.static.program_guard(self.main_program,
                                         self.startup_program):
            op = self.layer_help.append_op(type=self.op_type,
                                           inputs=self.input,
                                           outputs=self.output,
                                           attrs=self.attrs)

            prim_out = _orig2prim(op, *self.orig2prim_args)
            all_ops = [op.type for op in self.main_program.block(0).ops]

            self.assertEqual(sorted(all_ops), sorted(self.all_ops))
            prim_out = flatten(prim_out)
            for k, v in self.out_map.items():
                self.assertEqual(prim_out[k].shape, v.shape)


class TestSqrtOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'sqrt'
        X = paddle.static.data(name='X', shape=[7, 8], dtype='float64')

        self.input = {
            'X': X,
        }
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {}

        self.orig2prim_args = (X, )
        self.all_ops = ['sqrt', 'sqrt_p']
        # { prim_op_output_index: orig_op_output_var }
        self.out_map = {0: self.output['Out']}


class TestElementWiseMulOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'elementwise_mul'
        X = paddle.static.data(name='X', shape=[8, 8], dtype='float')
        Y = paddle.static.data(name='Y', shape=[8, 8], dtype='float')

        self.input = {'X': X, 'Y': Y}
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {}

        self.orig2prim_args = (X, Y)
        self.all_ops = ['elementwise_mul', 'mul_p']
        # { prim_op_output_index: orig_op_output_var }
        self.out_map = {0: self.output['Out']}


class TestMatmulV2Orig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'matmul_v2'
        X = paddle.static.data(name='X', shape=[3, 4], dtype='float')
        Y = paddle.static.data(name='Y', shape=[4, 3], dtype='float')

        self.input = {'X': X, 'Y': Y}
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {'trans_x': True, 'trans_y': True}

        self.orig2prim_args = (X, Y)
        self.all_ops = ['matmul_v2', 'transpose_p', 'transpose_p', 'matmul_p']
        self.out_map = {0: self.output['Out']}


class TestTanhOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'tanh'
        X = paddle.static.data(name='X', shape=[3, 4], dtype='float')

        self.input = {
            'X': X,
        }
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {}

        self.orig2prim_args = (X, )
        self.all_ops = ['tanh', 'tanh_p']
        self.out_map = {0: self.output['Out']}


class TestSinOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'sin'
        X = paddle.static.data(name='X', shape=[3, 4], dtype='float')

        self.input = {
            'X': X,
        }
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {}

        self.orig2prim_args = (X, )
        self.all_ops = ['sin', 'sin_p']
        self.out_map = {0: self.output['Out']}


class TestCosOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'cos'
        X = paddle.static.data(name='X', shape=[3, 4], dtype='float')

        self.input = {
            'X': X,
        }
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {}

        self.orig2prim_args = (X, )
        self.all_ops = ['cos', 'cos_p']
        self.out_map = {0: self.output['Out']}


class TestExpOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'exp'
        X = paddle.static.data(name='X', shape=[3, 4], dtype='float')

        self.input = {
            'X': X,
        }
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {}

        self.orig2prim_args = (X, )
        self.all_ops = ['exp', 'exp_p']
        self.out_map = {0: self.output['Out']}


class TestLogOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'log'
        X = paddle.static.data(name='X', shape=[3, 4], dtype='float')

        self.input = {
            'X': X,
        }
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {}

        self.orig2prim_args = (X, )
        self.all_ops = ['log', 'log_p']
        self.out_map = {0: self.output['Out']}


class TestReshape2Orig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'reshape2'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')

        self.input = {
            'X': X,
        }
        self.output = {
            'Out':
            X,
            'XShape':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {'shape': [6, 5]}

        self.orig2prim_args = (
            None,
            None,
            X,
        )
        self.all_ops = ['reshape2', 'reshape_p', 'fill_constant_p']
        # Do not checke XShape
        self.out_map = {0: self.output['Out']}


class TestConcatOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'concat'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        Y = paddle.static.data(name='Y', shape=[3, 6], dtype='int64')

        self.input = {
            'X': [X, Y],
        }
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {'axis': 0}

        self.orig2prim_args = (
            None,
            (X, Y),
        )
        self.all_ops = ['concat', 'concat_p']
        self.out_map = {0: self.output['Out']}


class TestSliceOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'slice'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')

        self.input = {
            'Input': X,
        }
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {
            'axes': [0],
            'starts': [1],
            'ends': [4],
        }

        self.orig2prim_args = (None, None, X, None, None)
        self.all_ops = ['slice', 'slice_select_p']
        self.out_map = {0: self.output['Out']}


class TestFillZerosLikeOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'fill_zeros_like'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')

        self.input = {
            'X': X,
        }
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {}

        self.orig2prim_args = (X, )
        self.all_ops = ['fill_zeros_like', 'fill_constant_p']
        self.out_map = {0: self.output['Out']}


class TestSumOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'sum'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        Y = paddle.static.data(name='Y', shape=[5, 6], dtype='int64')

        self.input = {'X': X, 'Y': Y}
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {}

        self.orig2prim_args = ((X, Y), )
        self.all_ops = ['sum', 'add_p']
        self.out_map = {0: self.output['Out']}


class TestPNormOrig2Prim1(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'p_norm'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')

        self.input = {
            'X': X,
        }
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {
            'porder': 1,
            'asvector': True,
        }

        self.orig2prim_args = (X, )
        self.all_ops = ['p_norm', 'reshape_p', 'sqrt_p', 'reduce_p', 'mul_p']
        self.out_map = {0: self.output['Out']}


class TestPNormOrig2Prim2(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'p_norm'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')

        self.input = {
            'X': X,
        }
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {
            'porder': 2,
            'asvector': True,
        }

        self.orig2prim_args = (X, )
        self.all_ops = ['p_norm', 'reshape_p', 'sqrt_p', 'reduce_p', 'mul_p']
        self.out_map = {0: self.output['Out']}


class TestIndexSelectOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'index_select'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        Index = paddle.static.data(name='Index', shape=[2], dtype='int32')

        self.input = {'X': X, 'Index': Index}
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {
            'dim': 0,
        }

        self.orig2prim_args = (
            Index,
            X,
        )
        self.all_ops = ['index_select', 'gather_p']
        self.out_map = {0: self.output['Out']}


class TestElementwiseSubOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'elementwise_sub'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int32')
        Y = paddle.static.data(name='Y', shape=[6], dtype='int32')

        self.input = {'X': X, 'Y': Y}
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {
            'dim': 0,
        }

        self.orig2prim_args = (
            X,
            Y,
        )
        self.all_ops = ['elementwise_sub', 'broadcast_p', 'sub_p']
        self.out_map = {0: self.output['Out']}


class TestScaleOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'scale'
        X = paddle.static.data(name='X', shape=[10, 7], dtype='int32')

        self.input = {
            'X': X,
        }
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {'scale': 2.0, 'bias': 1.0, 'bias_after_scale': True}

        self.orig2prim_args = (
            None,
            X,
        )
        self.all_ops = [
            'scale', 'fill_constant_p', 'fill_constant_p', 'mul_p', 'add_p'
        ]
        self.out_map = {0: self.output['Out']}


class TestAssignOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'assign'
        X = paddle.static.data(name='X', shape=[10, 7], dtype='int32')

        self.input = {
            'X': X,
        }
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {}

        self.orig2prim_args = (X, )
        self.all_ops = ['assign', 'fill_constant_p', 'add_p']
        self.out_map = {0: self.output['Out']}


class TestWhereOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'where'
        Cond = paddle.static.data(name='Condition', shape=[5, 6], dtype='bool')
        X = paddle.static.data(name='X', shape=[5, 6], dtype='float32')
        Y = paddle.static.data(name='Y', shape=[5, 6], dtype='float32')

        self.input = {'Condition': Cond, 'X': X, 'Y': Y}
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {}
        self.orig2prim_args = (Cond, X, Y)
        self.all_ops = ['where', 'select_p']
        self.out_map = {0: self.output['Out']}


class TestEqualOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'equal'
        X = paddle.static.data(name='X', shape=[5, 8], dtype='float')
        Y = paddle.static.data(name='Y', shape=[5, 8], dtype='float')

        self.input = {'X': X, 'Y': Y}
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype='bool')
        }
        self.attrs = {}
        self.orig2prim_args = (X, Y)
        self.all_ops = ['equal', 'eq_p']
        # { prim_op_output_index: orig_op_output_var }
        self.out_map = {0: self.output['Out']}


class TestPowOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'elementwise_pow'
        X = paddle.static.data(name='X', shape=[5, 8], dtype='float')
        Y = paddle.static.data(name='Y', shape=[5, 8], dtype='float')

        self.input = {'X': X, 'Y': Y}
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {}

        self.orig2prim_args = (X, Y)
        self.all_ops = ['elementwise_pow', 'pow_p']
        # { prim_op_output_index: orig_op_output_var }
        self.out_map = {0: self.output['Out']}


class TestMaxOrig2Prim(TestElementWiseAddOrig2Prim):

    def init_data(self):
        self.op_type = 'elementwise_max'
        X = paddle.static.data(name='X', shape=[5, 8], dtype='float')
        Y = paddle.static.data(name='Y', shape=[5, 8], dtype='float')

        self.input = {'X': X, 'Y': Y}
        self.output = {
            'Out':
            self.layer_help.create_variable_for_type_inference(dtype=X.dtype)
        }
        self.attrs = {}

        self.orig2prim_args = (X, Y)
        self.all_ops = ['elementwise_max', 'max_p']
        # { prim_op_output_index: orig_op_output_var }
        self.out_map = {0: self.output['Out']}


if __name__ == '__main__':
    unittest.main()
