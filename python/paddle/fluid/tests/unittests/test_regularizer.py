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

import unittest

import paddle.fluid.framework as framework
import paddle.fluid.optimizer as optimizer
import paddle.fluid.regularizer as regularizer
from paddle.fluid.backward import append_backward


class TestL2DecayRegularizer(unittest.TestCase):
    def test_l2decay_regularizer(self):
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            regularizer=regularizer.L2DecayRegularizer(0.5))
        self.assertTrue(mul_x.regularizer is not None)
        self.assertTrue(
            isinstance(mul_x.regularizer, regularizer.L2DecayRegularizer))
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
        block.append_op(
            type="mul",
            inputs={"X": mul_x,
                    "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1})
        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out")
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out})
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        count_ops = len(block.ops)
        params_grads = optimizer.append_regularization_ops(params_grads)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(block.ops), count_ops + 2)
        self.assertEqual(block.ops[-1].type, 'elementwise_add')
        self.assertEqual(block.ops[-2].type, 'scale')


class TestL1DecayRegularizer(unittest.TestCase):
    def test_l2decay_regularizer(self):
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            regularizer=regularizer.L1DecayRegularizer(0.5))
        self.assertTrue(mul_x.regularizer is not None)
        self.assertTrue(
            isinstance(mul_x.regularizer, regularizer.L1DecayRegularizer))
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
        block.append_op(
            type="mul",
            inputs={"X": mul_x,
                    "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1})
        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out")
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out})
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        count_ops = len(block.ops)
        params_grads = optimizer.append_regularization_ops(params_grads)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(block.ops), count_ops + 3)
        self.assertEqual(block.ops[-1].type, 'elementwise_add')
        self.assertEqual(block.ops[-2].type, 'scale')
        self.assertEqual(block.ops[-3].type, 'sign')


if __name__ == '__main__':
    unittest.main()
