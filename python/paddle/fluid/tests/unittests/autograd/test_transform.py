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
import numpy as np

import paddle
from paddle.incubate.autograd.primx import Transform, orig2prim, prim2orig
from paddle.fluid.layers.utils import flatten

paddle.enable_static()


class TestAutoGradTransformForAdd(unittest.TestCase):

    def setUp(self):
        self.main_program = paddle.static.Program()
        self.startup_program = paddle.static.Program()

        with paddle.static.program_guard(self.main_program,
                                         self.startup_program):
            self.init_data()

    def init_data(self):
        # { input_index: input_shape }
        self.xs_shape_map = {0: (20, 40), 1: (20, 40)}
        # { output_index: output_shape }
        self.ys_shape_map = {0: (20, 40)}
        X0 = paddle.static.data(name='X0',
                                shape=self.xs_shape_map[0],
                                dtype='float32')
        X0.stop_gradient = False
        X1 = paddle.static.data(name='X1',
                                shape=self.xs_shape_map[1],
                                dtype='float32')
        X1.stop_gradient = False

        A = paddle.tanh(X0)
        B = paddle.tanh(X1)
        Y = paddle.add(A, B)

        self.orig_xs = [X0, X1]
        self.orig_ys = [
            Y,
        ]

        self.orig_ops = ['tanh', 'tanh', 'elementwise_add']
        self.orig2prim_ops = ['tanh_p', 'tanh_p', 'add_p']
        self.linearize_ops = self.orig2prim_ops + [
            # call fill_const() in linearize() function
            'fill_constant_p',
            'fill_constant_p',
            # linearized op
            'mul_p',
            'sub_p',
            'fill_constant_p',
            'mul_p',
            'mul_p',
            'sub_p',
            'fill_constant_p',
            'mul_p',
            'add_p',
        ]
        self.transpose_ops = self.orig2prim_ops + [
            # call fill_const() in transpose() function
            'fill_constant_p',
            # linearized op after remove path
            'fill_constant_p',
            'fill_constant_p',
            'mul_p',
            'sub_p',
            'fill_constant_p',
            'mul_p',
            'sub_p',
            'fill_constant_p',
            # transposed op
            'mul_p',
            'mul_p'
        ]
        self.prim2orig_ops_with_blacklist = [
            'tanh', 'tanh', 'add_p', 'fill_constant', 'fill_constant',
            'fill_constant', 'elementwise_mul', 'sub_p', 'fill_constant',
            'elementwise_mul', 'sub_p', 'fill_constant', 'elementwise_mul',
            'elementwise_mul'
        ]
        self.prim2orig_ops = [
            'tanh', 'tanh', 'elementwise_add', 'fill_constant', 'fill_constant',
            'fill_constant', 'elementwise_mul', 'elementwise_sub',
            'fill_constant', 'elementwise_mul', 'elementwise_sub',
            'fill_constant', 'elementwise_mul', 'elementwise_mul'
        ]

    def test_run(self):
        # Must using with program_guard(), otherwise prim ops will append other block
        with paddle.static.program_guard(self.main_program,
                                         self.startup_program):
            ad = Transform(self.main_program.block(0))
            orig_ops = [op.type for op in self.main_program.block(0).ops]
            self.assertEqual(sorted(orig_ops), sorted(self.orig_ops))

            # Test orig2prim
            orig2prim(block=self.main_program.block(0))
            orig2prim_ops = [op.type for op in self.main_program.block(0).ops]
            self.assertEqual(sorted(orig2prim_ops), sorted(self.orig2prim_ops))

            # Test linearize
            xs_dot, ys_dot = ad.linearize(self.orig_xs, self.orig_ys)
            linearize_ops = [op.type for op in self.main_program.block(0).ops]
            self.assertEqual(sorted(linearize_ops), sorted(self.linearize_ops))
            flatten_xs_dot = flatten(xs_dot)
            for k, v in self.xs_shape_map.items():
                self.assertEqual(flatten_xs_dot[k].shape, v)
            flatten_ys_dot = flatten(ys_dot)
            for k, v in self.ys_shape_map.items():
                self.assertEqual(flatten_ys_dot[k].shape, v)

            # Test transpose
            ys_bar, xs_bar = ad.transpose(ys_dot, xs_dot, retain_fwd=False)
            transpose_ops = [op.type for op in self.main_program.block(0).ops]
            self.assertEqual(sorted(transpose_ops), sorted(self.transpose_ops))
            flatten_xs_bar = flatten(xs_bar)
            for k, v in self.xs_shape_map.items():
                # There may be None in the result of transpose like gather op
                if flatten_xs_bar[k] is not None:
                    self.assertEqual(flatten_xs_bar[k].shape, v)
            flatten_ys_bar = flatten(ys_bar)
            for k, v in self.ys_shape_map.items():
                self.assertEqual(flatten_ys_bar[k].shape, v)

            # Test prim2orig with blacklist
            prim2orig(block=self.main_program.block(0),
                      blacklist=['add_p', 'sub_p'])
            prim2orig_ops = [op.type for op in self.main_program.block(0).ops]
            self.assertEqual(sorted(prim2orig_ops),
                             sorted(self.prim2orig_ops_with_blacklist))

            # Test prim2orig
            prim2orig(block=self.main_program.block(0))
            prim2orig_ops = [op.type for op in self.main_program.block(0).ops]
            self.assertEqual(sorted(prim2orig_ops), sorted(self.prim2orig_ops))


class TestAutoGradTransformForMatmul(TestAutoGradTransformForAdd):

    def init_data(self):
        # { input_index: input_shape }
        self.xs_shape_map = {0: (100, 2), 1: (5, 2)}
        # { output_index: output_shape }
        self.ys_shape_map = {0: (100, 5)}
        X0 = paddle.static.data('X0',
                                shape=self.xs_shape_map[0],
                                dtype='float32')
        X0.stop_gradient = False
        X1 = paddle.static.data('X1',
                                shape=self.xs_shape_map[1],
                                dtype='float32')
        X1.stop_gradient = False

        A = paddle.reshape(X1, [2, 5])
        B = paddle.scale(A, scale=2.0, bias=2.0)
        Y = paddle.matmul(X0, B)

        self.orig_xs = [X0, X1]
        self.orig_ys = [
            Y,
        ]

        self.orig_ops = ['reshape2', 'scale', 'matmul_v2']
        self.orig2prim_ops = [
            'reshape_p', 'fill_constant_p', 'fill_constant_p',
            'fill_constant_p', 'mul_p', 'add_p', 'matmul_p'
        ]
        self.linearize_ops = self.orig2prim_ops + [
            # call fill_const() in linearize() function
            'fill_constant_p',
            'fill_constant_p',
            # linearized op
            'reshape_p',
            'mul_p',
            # 'mul_p', # JVP rules handle `None` input, some op will not be appended
            # 'add_p',
            # 'add_p',
            'matmul_p',
            'matmul_p',
            'add_p'
        ]
        self.transpose_ops = self.orig2prim_ops + [
            # call fill_const() in transpose() function
            'fill_constant_p',
            # linearized op after remove path
            'fill_constant_p',
            'fill_constant_p',
            'mul_p',
            # transposed op
            'transpose_p',
            'matmul_p',
            'transpose_p',
            'matmul_p',
            # 'mul_p',
            'reshape_p',
        ]

        self.prim2orig_ops_with_blacklist = [
            'reshape2',
            'fill_constant',
            'fill_constant',
            'fill_constant',
            'elementwise_mul',
            'add_p',
            'matmul_v2',
            'fill_constant',
            'fill_constant',
            'fill_constant',
            'elementwise_mul',
            'transpose2',
            'matmul_v2',
            'transpose2',
            'matmul_v2',
            # 'elementwise_mul',
            'reshape2',
        ]

        self.prim2orig_ops = [
            'reshape2',
            'fill_constant',
            'fill_constant',
            'fill_constant',
            'elementwise_mul',
            'elementwise_add',
            'matmul_v2',
            'fill_constant',
            'fill_constant',
            'fill_constant',
            'elementwise_mul',
            'transpose2',
            'matmul_v2',
            'transpose2',
            'matmul_v2',
            # 'elementwise_mul',
            'reshape2',
        ]


class TestAutoGradTransformForIndexSelect(TestAutoGradTransformForAdd):

    def init_data(self):
        # { input_index: input_shape }
        self.xs_shape_map = {0: (7, 8, 9), 1: (8, 1), 2: (7, 8, 9), 3: (3, )}
        # { output_index: output_shape }
        self.ys_shape_map = {0: (3, 16, 9)}

        X0 = paddle.static.data('X0',
                                shape=self.xs_shape_map[0],
                                dtype='float32')
        X0.stop_gradient = False
        X1 = paddle.static.data('X1',
                                shape=self.xs_shape_map[1],
                                dtype='float32')
        X1.stop_gradient = False
        X2 = paddle.static.data('X2',
                                shape=self.xs_shape_map[2],
                                dtype='float32')
        X2.stop_gradient = False
        X3 = paddle.static.data('X3', shape=self.xs_shape_map[3], dtype='int32')
        X3.stop_gradient = False

        A = paddle.add(X0, X1)  # (7, 8, 9)
        B = paddle.norm(x=A, p=2)  # (1, )
        C = paddle.subtract(X2, B)  # (7, 8, 9)
        D = paddle.concat(x=(A, C), axis=1)  # (7, 16, 9)
        Y = paddle.index_select(D, X3, axis=0)  # (3, 16, 9)

        self.orig_xs = [X0, X1, X2, X3]
        self.orig_ys = [
            Y,
        ]
        self.orig_ops = [
            'elementwise_add', 'p_norm', 'elementwise_sub', 'concat',
            'index_select'
        ]
        self.orig2prim_ops = [
            'broadcast_p', 'add_p', 'reshape_p', 'mul_p', 'reduce_sum_p',
            'sqrt_p', 'broadcast_p', 'sub_p', 'concat_p', 'gather_p'
        ]
        self.linearize_ops = self.orig2prim_ops + [
            # call fill_const() in linearize() function
            'fill_constant_p',
            'fill_constant_p',
            'fill_constant_p',
            'fill_constant_p',
            # linearized op
            'broadcast_p',
            'add_p',
            'reshape_p',
            'mul_p',
            'mul_p',
            'add_p',
            'reduce_sum_p',
            'fill_constant_p',  # 'sqrt_p', Will not append sqrt_p op when apply JVP for sqrt_p
            'mul_p',
            'div_p',
            'broadcast_p',
            'sub_p',
            'concat_p',
            'gather_p'
        ]
        self.transpose_ops = self.orig2prim_ops + [
            # call fill_const() in transpose() function
            'fill_constant_p',
            # linearized op after remove path
            'fill_constant_p',
            'fill_constant_p',
            'fill_constant_p',
            'fill_constant_p',
            'fill_constant_p',
            'mul_p',
            # transposed op
            'reduce_sum_p',
            'reshape_p',
            'reshape_p',
            'mul_p',
            'mul_p',
            'reshape_p',
            'broadcast_p',
            'div_p',
            'reduce_sum_p',
            'reshape_p',
            'fill_constant_p',
            'sub_p',
            'split_p',
            'fill_constant_p',
            'scatter_add_p',
            'add_p',  # The output of the op is used by multiple subsequent ops
            'add_p',
        ]

        self.prim2orig_ops_with_blacklist = [
            'expand_v2', 'add_p', 'reshape2', 'elementwise_mul', 'reduce_sum',
            'sqrt', 'expand_v2', 'sub_p', 'concat', 'gather', 'fill_constant',
            'fill_constant', 'fill_constant', 'fill_constant', 'fill_constant',
            'fill_constant', 'elementwise_mul', 'reduce_sum', 'reshape2',
            'reshape2', 'elementwise_mul', 'elementwise_mul', 'reshape2',
            'expand_v2', 'elementwise_div', 'reduce_sum', 'reshape2',
            'fill_constant', 'sub_p', 'split', 'fill_constant', 'fill_any_like',
            'add_p', 'scatter', 'elementwise_add', 'add_p'
        ]

        self.prim2orig_ops = [
            'expand_v2', 'elementwise_add', 'reshape2', 'elementwise_mul',
            'reduce_sum', 'sqrt', 'expand_v2', 'elementwise_sub', 'concat',
            'gather', 'fill_constant', 'fill_constant', 'fill_constant',
            'fill_constant', 'fill_constant', 'fill_constant',
            'elementwise_mul', 'reduce_sum', 'reshape2', 'reshape2',
            'elementwise_mul', 'elementwise_mul', 'reshape2', 'expand_v2',
            'elementwise_div', 'reduce_sum', 'reshape2', 'fill_constant',
            'elementwise_sub', 'split', 'fill_constant', 'fill_any_like',
            'elementwise_add', 'scatter', 'elementwise_add', 'elementwise_add'
        ]


if __name__ == '__main__':
    unittest.main()
