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

from __future__ import print_function

import unittest

from paddle.fluid.framework import Program
from paddle.fluid.executor import Executor
from paddle.fluid.backward import append_backward
import numpy as np
import paddle.fluid.core as core


class RNNMemoryHelperOpTest(unittest.TestCase):
    def setUp(self):
        self.program = Program()
        self.place = core.CPUPlace()

        self.X = self.program.global_block().create_var(
            name='X', shape=[2, 3], dtype='float32')
        self.Out = self.program.global_block().create_var(
            name='Out', shape=[2, 3], dtype='float32')
        self.program.global_block().append_op(
            type='rnn_memory_helper',
            inputs={"X": self.X},
            outputs={"Out": self.Out},
            attrs={})

    def test_forward(self):
        x_np = np.random.normal(size=(2, 3)).astype("float32")
        self.feed_map = {'X': x_np}
        self.fetch_list = [self.Out]
        exe = Executor(self.place)
        out = exe.run(self.program,
                      feed=self.feed_map,
                      fetch_list=self.fetch_list)
        self.assertTrue(np.allclose(out[0], x_np, rtol=1e-5))


class RNNMemoryHelperGradOpTest(unittest.TestCase):
    def setup(self, input_names):
        self.place = core.CPUPlace()

        fwd_program = Program()
        self.program = Program()

        self.input_names = input_names
        self.input_vars = {
            name: self.program.global_block().create_var(
                name=name, shape=[2, 3], dtype='float32')
            for name in self.input_names
        }

        self.output_names = ['X@GRAD']
        self.output_vars = {
            name: self.program.global_block().create_var(
                name=name, shape=[2, 3], dtype='float32')
            for name in self.output_names
        }

        fwd_op = fwd_program.global_block().append_op(
            type='rnn_memory_helper',
            inputs={"X": self.input_vars['X']},
            outputs={"Out": self.input_vars['Out']},
            attrs={})

        # construct grad operator by forward operator
        grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(fwd_op.desc,
                                                                  set(), [])
        grad_op_desc = grad_op_desc_list[0]
        grad_block = self.program.global_block()
        op = grad_block.desc.append_op()
        op.copy_from(grad_op_desc)

        self.program._sync_with_cpp()

    def setup_with_input(self):
        self.setup(input_names=['X', 'Out', 'Out@GRAD'])

    def setup_without_input(self):
        self.setup(input_names=['X', 'Out'])

    def test_backward(self):
        self.setup_with_input()
        self.feed_map = {
            name: np.random.normal(size=(2, 3)).astype("float32")
            for name in self.input_names
        }
        self.fetch_list = [self.output_vars['X@GRAD']]

        exe = Executor(self.place)
        out = exe.run(self.program,
                      feed=self.feed_map,
                      fetch_list=self.fetch_list)
        np.isclose(out[0], self.feed_map['Out@GRAD'], rtol=1e-5)

    def test_backward_without_Input(self):
        self.setup_without_input()
        self.feed_map = {
            name: np.random.normal(size=(2, 3)).astype("float32")
            for name in ['X', 'Out']
        }
        self.fetch_list = [self.output_vars['X@GRAD']]

        exe = Executor(self.place)
        out = exe.run(self.program,
                      feed=self.feed_map,
                      fetch_list=self.fetch_list)
        self.assertTrue(
            np.allclose(
                out[0], np.zeros(shape=(2, 3)).astype("float32"), rtol=1e-5))


if __name__ == '__main__':
    unittest.main()
