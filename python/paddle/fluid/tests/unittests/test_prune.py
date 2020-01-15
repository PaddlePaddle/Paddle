#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.framework as framework
import paddle.compat as cpt


class TestPrune(unittest.TestCase):
    def net(self):
        x = fluid.layers.data(name='x', shape=[2], dtype='float32')
        label = fluid.layers.data(name="label", shape=[1], dtype="int64")
        y = fluid.layers.fc(input=[x], size=2, act="softmax")
        loss = fluid.layers.cross_entropy(input=y, label=label)
        loss = fluid.layers.mean(x=loss)
        return x, y, label, loss

    def test_prune_with_input(self):
        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()
        with fluid.program_guard(program, startup_program):
            (x, y, label, loss) = self.net()
        self.assertEqual(len(block.ops), 5)
        self.assertEqual([op.type for op in block.ops], [
            "mul", "elementwise_add", "softmax", "cross_entropy2", "mean"
        ])
        pruned_program = program._prune_with_input(
            feeded_var_names=[y.name, label.name], targets=[loss])
        self.assertEqual(len(pruned_program.global_block().ops), 2)
        self.assertEqual([op.type for op in pruned_program.global_block().ops],
                         ["cross_entropy2", "mean"])

    def test_prune(self):
        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()
        with fluid.program_guard(program, startup_program):
            (x, y, label, loss) = self.net()
        self.assertEqual(len(block.ops), 5)
        self.assertEqual([op.type for op in block.ops], [
            "mul", "elementwise_add", "softmax", "cross_entropy2", "mean"
        ])
        pruned_program = program._prune(targets=[loss])
        self.assertEqual(len(pruned_program.global_block().ops), 5)
        self.assertEqual(
            [op.type for op in pruned_program.global_block().ops],
            ["mul", "elementwise_add", "softmax", "cross_entropy2", "mean"])

    def test_prune_target_not_list(self):
        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()
        with fluid.program_guard(program, startup_program):
            (x, y, label, loss) = self.net()
        self.assertEqual(len(block.ops), 5)
        self.assertEqual([op.type for op in block.ops], [
            "mul", "elementwise_add", "softmax", "cross_entropy2", "mean"
        ])
        pruned_program = program._prune(targets=loss)
        self.assertEqual(len(pruned_program.global_block().ops), 5)
        self.assertEqual(
            [op.type for op in pruned_program.global_block().ops],
            ["mul", "elementwise_add", "softmax", "cross_entropy2", "mean"])

    def test_prune_target_none(self):
        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()
        with fluid.program_guard(program, startup_program):
            (x, y, label, loss) = self.net()
        self.assertEqual(len(block.ops), 5)
        self.assertEqual([op.type for op in block.ops], [
            "mul", "elementwise_add", "softmax", "cross_entropy2", "mean"
        ])
        try:
            pruned_program = program._prune(targets=None)
        except ValueError as e:
            self.assertEqual(
                "All targets of prune() can only be Variable or Operator.",
                cpt.get_exception_message(e))


if __name__ == '__main__':
    unittest.main()
