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

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.optimizer as optimizer
from paddle.fluid.framework import Program, program_guard
from paddle.fluid.transpiler import memory_optimize


def _get_vars(prog):
    assert (isinstance(prog, Program))
    all_vars = set()
    for op in prog.global_block().ops:
        all_vars.update(op.input_arg_names)
        all_vars.update(op.output_arg_names)
    return all_vars


class TestControlFlowGraph(unittest.TestCase):
    def setUp(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            x = layers.data(name='x', shape=[13], dtype='float32')
            y_predict = layers.fc(input=x, size=1, act=None)
            y = layers.data(name='y', shape=[1], dtype='float32')
            cost = layers.square_error_cost(input=y_predict, label=y)
            avg_cost = layers.mean(cost)
            opt = optimizer.SGD(learning_rate=0.001)
            opt = opt.minimize(avg_cost)

        self.program = program

    def test_control_flow_graph(self):
        result_program = self.program.clone()
        memory_optimize(self.program)
        old_vars = _get_vars(self.program)
        new_vars = _get_vars(result_program)
        self.assertTrue(old_vars != new_vars)


class TestMemoryTranspiler2(unittest.TestCase):
    def setUp(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            x = layers.data(name='x', shape=[13], dtype='float32')
            fc = layers.fc(input=x, size=10, act=None)
            reshape = layers.reshape(x=fc, shape=[-1, 2, 5])
            fc = layers.reshape(x=reshape, shape=[-1, 5, 2])
            y_predict = layers.fc(input=fc, size=1, act=None)
            y = layers.data(name='y', shape=[1], dtype='float32')
            cost = layers.square_error_cost(input=y_predict, label=y)
            avg_cost = layers.mean(cost)
            opt = optimizer.SGD(learning_rate=0.001)
            opt.minimize(avg_cost)
        self.skip_set = set([cost.name, fc.name])
        self.program = program

    def test_inplace_ops(self):
        result_program = self.program.clone()
        memory_optimize(self.program)
        old_vars = _get_vars(self.program)
        new_vars = _get_vars(result_program)
        self.assertTrue(old_vars != new_vars)

    def test_skip_opt(self):
        result_program = self.program.clone()
        memory_optimize(self.program, skip_opt_set=self.skip_set)
        old_vars = _get_vars(self.program)
        new_vars = _get_vars(result_program)
        self.assertTrue(old_vars != new_vars)


class TestMemoryTranspiler3(unittest.TestCase):
    def setUp(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            word = fluid.layers.data(name='word', shape=[1], dtype='int64')
            emb = [
                fluid.layers.embedding(
                    word, size=[65536, 256], param_attr='emb') for _ in range(6)
            ]

            left = emb.pop(0)
            while len(emb) != 0:
                right = emb.pop(0)
                left = fluid.layers.concat([left, right])
            emb = fluid.layers.mean(left)
            fluid.backward.append_backward(emb)
        self.program = program

    def test_cascade_reuse(self):
        block = self.program.block(0)
        # variable reuse in programdesc
        # TODO(dzhwinter): confirm cascade strategy. disable temporialy
        self.assertTrue("concat_4.tmp_0@GRAD" in block.vars)
        # self.assertTrue("concat_3.tmp_0@GRAD" not in block.vars)
        # self.assertTrue("concat_2.tmp_0@GRAD" not in block.vars)
        # self.assertTrue("concat_1.tmp_0@GRAD" not in block.vars)
        # self.assertTrue("concat_0.tmp_0@GRAD" not in block.vars)


if __name__ == "__main__":
    unittest.main()
