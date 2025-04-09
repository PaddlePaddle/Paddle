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

import numpy as np

import paddle
from paddle import base
from paddle.base import core, framework
from paddle.base.backward import append_backward
from paddle.base.framework import (
    Program,
    program_guard,
)

paddle.enable_static()


class TestOptimizer(unittest.TestCase):
    def test_sgd_optimizer(self):
        def check_sgd_optimizer(optimizer_attr):
            init_program = framework.Program()
            program = framework.Program()
            block = program.global_block()
            mul_x = block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                name="mul.x",
                optimize_attr=optimizer_attr,
            )
            mul_y = block.create_var(
                dtype="float32", shape=[10, 8], name="mul.y"
            )
            mul_out = block.create_var(
                dtype="float32", shape=[5, 8], name="mul.out"
            )
            mean_out = block.create_var(
                dtype="float32", shape=[1], name="mean.out"
            )
            block.append_op(
                type="mul",
                inputs={"X": mul_x, "Y": mul_y},
                outputs={"Out": mul_out},
                attrs={"x_num_col_dims": 1},
            )
            block.append_op(
                type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out}
            )
            sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            opts, _ = sgd_optimizer.minimize(mean_out, init_program)
            return opts

        opts = check_sgd_optimizer({'learning_rate': 1.1})
        self.assertEqual(len(opts), 2)
        self.assertEqual([op.type for op in opts], ["scale", "sgd"])

        opts = check_sgd_optimizer({'learning_rate': 1.0})
        self.assertEqual(len(opts), 1)
        self.assertEqual([op.type for op in opts], ["sgd"])


class TestOptimizerBackwardApplygrad(unittest.TestCase):
    def test_sgd_optimizer(self):
        def check_sgd_optimizer(optimizer_attr):
            init_program = framework.Program()
            program = framework.Program()
            block = program.global_block()
            mul_x = block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                name="mul.x",
                optimize_attr=optimizer_attr,
            )
            mul_y = block.create_var(
                dtype="float32", shape=[10, 8], name="mul.y"
            )
            mul_out = block.create_var(
                dtype="float32", shape=[5, 8], name="mul.out"
            )
            mean_out = block.create_var(
                dtype="float32", shape=[1], name="mean.out"
            )
            block.append_op(
                type="mul",
                inputs={"X": mul_x, "Y": mul_y},
                outputs={"Out": mul_out},
                attrs={"x_num_col_dims": 1},
            )
            block.append_op(
                type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out}
            )
            sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            with framework.program_guard(program, init_program):
                p_g = sgd_optimizer.backward(mean_out)
                opts = sgd_optimizer.apply_gradients(p_g)
            return opts

        opts = check_sgd_optimizer({'learning_rate': 1.1})
        self.assertEqual(len(opts), 2)
        self.assertEqual([op.type for op in opts], ["scale", "sgd"])

        opts = check_sgd_optimizer({'learning_rate': 1.0})
        self.assertEqual(len(opts), 1)
        self.assertEqual([op.type for op in opts], ["sgd"])


class TestMomentumOptimizer(unittest.TestCase):
    class MockMomentum(paddle.optimizer.Momentum):
        def get_accumulators(self):
            return self._accumulators

        def get_velocity_str(self):
            return self._velocity_acc_str

    def test_vanilla_momentum_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            name="mul.x",
            optimize_attr={'learning_rate': 1.1},
        )
        mul_y = block.create_var(dtype="float32", shape=[10, 8], name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], name="mul.out"
        )
        block.append_op(
            type="mul",
            inputs={"X": mul_x, "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1},
        )
        learning_rate = 0.01
        momentum_optimizer = self.MockMomentum(
            learning_rate=learning_rate, momentum=0.2
        )
        mean_out = block.create_var(dtype="float32", shape=[1], name="mean.out")
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out}
        )
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(momentum_optimizer.get_accumulators()), 0)
        with framework.program_guard(program, init_program):
            opts = momentum_optimizer.apply_gradients(params_grads)
        self.assertEqual(len(opts), 2)
        sgd_op = opts[-1]
        self.assertEqual([op.type for op in opts], ["scale", "momentum"])
        self.assertFalse(sgd_op.attr('use_nesterov'))

        # Check accumulators
        accumulators = momentum_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 1)
        self.assertTrue(momentum_optimizer.get_velocity_str() in accumulators)
        velocity_acc = accumulators[momentum_optimizer.get_velocity_str()]
        self.assertEqual(len(velocity_acc), 1)
        self.assertTrue(mul_x.name in velocity_acc)

        # Check init_program
        init_ops = init_program.global_block().ops
        self.assertEqual(len(init_ops), 2)
        self.assertEqual(init_ops[1].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[1].attr('value'), 0.0)

    def test_nesterov_momentum_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            name="mul.x",
            optimize_attr={'learning_rate': 1.1},
        )
        mul_y = block.create_var(dtype="float32", shape=[10, 8], name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], name="mul.out"
        )
        block.append_op(
            type="mul",
            inputs={"X": mul_x, "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1},
        )
        mean_out = block.create_var(dtype="float32", shape=[1], name="mean.out")
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out}
        )
        learning_rate = 0.01
        momentum_optimizer = self.MockMomentum(
            learning_rate=learning_rate, momentum=0.2, use_nesterov=True
        )
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(momentum_optimizer.get_accumulators()), 0)
        with framework.program_guard(program, init_program):
            opts = momentum_optimizer.apply_gradients(params_grads)
        self.assertEqual(len(opts), 2)
        sgd_op = opts[-1]
        self.assertEqual([op.type for op in opts], ["scale", "momentum"])
        self.assertTrue(sgd_op.attr('use_nesterov'))

        # Check accumulators
        accumulators = momentum_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 1)
        self.assertTrue(momentum_optimizer.get_velocity_str() in accumulators)
        velocity_acc = accumulators[momentum_optimizer.get_velocity_str()]
        self.assertEqual(len(velocity_acc), 1)
        self.assertTrue(mul_x.name in velocity_acc)

        # Check init_program
        init_ops = init_program.global_block().ops
        self.assertEqual(len(init_ops), 2)
        self.assertEqual(init_ops[1].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[1].attr('value'), 0.0)


class TestAdamOptimizer(unittest.TestCase):
    class MockAdam(paddle.optimizer.Adam):
        def get_accumulators(self):
            return self._accumulators

        def get_moment1_str(self):
            return self._moment1_acc_str

        def get_moment2_str(self):
            return self._moment2_acc_str

    def test_adam_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            name="mul.x",
            optimize_attr={'learning_rate': 1.1},
        )
        mul_y = block.create_var(dtype="float32", shape=[10, 8], name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], name="mul.out"
        )
        block.append_op(
            type="mul",
            inputs={"X": mul_x, "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1},
        )
        mean_out = block.create_var(dtype="float32", shape=[1], name="mean.out")
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out}
        )
        learning_rate = 0.01
        adam_optimizer = self.MockAdam(
            learning_rate=learning_rate, beta1=0.9, beta2=0.999
        )
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(adam_optimizer.get_accumulators()), 0)
        with framework.program_guard(program, init_program):
            opts = adam_optimizer.apply_gradients(params_grads)
        self.assertEqual(len(opts), 2)
        self.assertEqual([op.type for op in opts], ["scale", "adam"])

        # Check accumulators
        accumulators = adam_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 4)
        self.assertTrue(adam_optimizer.get_moment1_str() in accumulators)
        self.assertTrue(adam_optimizer.get_moment2_str() in accumulators)
        moment1_acc = accumulators[adam_optimizer.get_moment1_str()]
        moment2_acc = accumulators[adam_optimizer.get_moment2_str()]
        self.assertEqual(len(moment1_acc), 1)
        self.assertEqual(len(moment2_acc), 1)
        self.assertTrue(mul_x.name in moment1_acc)
        self.assertTrue(mul_x.name in moment2_acc)

        # Check init_program
        init_ops = init_program.global_block().ops
        self.assertEqual(len(init_ops), 5)
        self.assertEqual(init_ops[-1].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)


class TestRecomputeOptimizer(unittest.TestCase):
    def net(self, return_input=False, with_dropout=False, with_seed=False):
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32", shape=[5, 10], name="mul.x"
        )
        mul_y = block.create_var(dtype="float32", shape=[10, 8], name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], name="mul.out"
        )

        if with_dropout is True:
            mul_out_drop = block.create_var(
                dtype="float32",
                shape=[5, 8],
                name="mul.out.dropout",
            )
            mul_out_mask = block.create_var(
                dtype="uint8", shape=[5, 8], name="mul.out.mask"
            )
            if with_seed is True:
                seed_out = block.create_var(
                    dtype="int32", shape=[1], name="seed.out"
                )

        b1 = block.create_parameter(dtype="float32", shape=[5, 8], name="b1")
        b1_out = block.create_var(dtype="float32", shape=[5, 8], name="b1_out")
        b2 = block.create_parameter(dtype="float32", shape=[5, 8], name="b2")
        b2_out = block.create_var(dtype="float32", shape=[5, 8], name="b2_out")
        mean_out = block.create_var(dtype="float32", shape=[1], name="mean.out")
        block.append_op(
            type="mul",
            inputs={"X": mul_x, "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1},
        )

        if with_dropout is True:
            dropout_inputs = {'X': [mul_out]}
            if with_seed is True:
                block.append_op(
                    type='seed',
                    outputs={'Out': seed_out},
                    attrs={
                        'deterministic': True,
                        'rng_name': 'rng0',
                        'force_cpu': True,
                    },
                )
                dropout_inputs = {'X': [mul_out], 'Seed': [seed_out]}

            block.append_op(
                type='dropout',
                inputs=dropout_inputs,
                outputs={'Out': [mul_out_drop], 'Mask': [mul_out_mask]},
                attrs={
                    'dropout_prob': 0.5,
                },
            )
            block.append_op(
                type="elementwise_add",
                inputs={"X": mul_out_drop, "Y": b1},
                outputs={"Out": b1_out},
            )
        else:
            block.append_op(
                type="elementwise_add",
                inputs={"X": mul_out, "Y": b1},
                outputs={"Out": b1_out},
            )

        block.append_op(
            type="elementwise_add",
            inputs={"X": b1_out, "Y": b2},
            outputs={"Out": b2_out},
        )
        block.append_op(
            type="mean", inputs={"X": b2_out}, outputs={"Out": mean_out}
        )

        if return_input:
            return mul_x, mul_out, b1_out, b2_out, mean_out
        return mul_out, b1_out, b2_out, mean_out

    def test_no_checkpoint(self):
        mul_out, b1_out, b2_out, mean_out = self.net()
        self.assertEqual(len(mean_out.block.ops), 4)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            ["mul", "elementwise_add", "elementwise_add", "mean"],
        )
        sgd_optimizer = paddle.optimizer.SGD(learning_rate=1.0)
        recompute_optimizer = paddle.incubate.optimizer.RecomputeOptimizer(
            sgd_optimizer
        )
        recompute_optimizer._set_checkpoints([])
        opts, params_grads = recompute_optimizer.minimize(mean_out)

        self.assertEqual(len(mean_out.block.ops), 12)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            [
                "mul",
                "elementwise_add",
                "elementwise_add",
                "mean",
                "fill_constant",
                "mean_grad",
                "elementwise_add_grad",
                "elementwise_add_grad",
                "mul_grad",
                "sgd",
                "sgd",
                "sgd",
            ],
        )

    def test_one_checkpoint(self):
        mul_out, b1_out, b2_out, mean_out = self.net()
        self.assertEqual(len(mean_out.block.ops), 4)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            ["mul", "elementwise_add", "elementwise_add", "mean"],
        )
        sgd_optimizer = paddle.optimizer.SGD(learning_rate=1.0)
        recompute_optimizer = paddle.incubate.optimizer.RecomputeOptimizer(
            sgd_optimizer
        )
        recompute_optimizer._set_checkpoints([b1_out])
        opts, params_grads = recompute_optimizer.minimize(mean_out)

        self.assertEqual(len(mean_out.block.ops), 13)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            [
                "mul",
                "elementwise_add",
                "elementwise_add",
                "mean",
                "fill_constant",
                "mean_grad",
                "elementwise_add_grad",
                "mul",
                "elementwise_add_grad",
                "mul_grad",
                "sgd",
                "sgd",
                "sgd",
            ],
        )

    def test_str_checkpoints(self):
        mul_out, b1_out, b2_out, mean_out = self.net()
        self.assertEqual(len(mean_out.block.ops), 4)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            ["mul", "elementwise_add", "elementwise_add", "mean"],
        )
        sgd_optimizer = paddle.optimizer.SGD(learning_rate=1.0)
        recompute_optimizer = paddle.incubate.optimizer.RecomputeOptimizer(
            sgd_optimizer
        )
        recompute_optimizer._set_checkpoints([b1_out.name])
        opts, params_grads = recompute_optimizer.minimize(mean_out)

        self.assertEqual(len(mean_out.block.ops), 13)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            [
                "mul",
                "elementwise_add",
                "elementwise_add",
                "mean",
                "fill_constant",
                "mean_grad",
                "elementwise_add_grad",
                "mul",
                "elementwise_add_grad",
                "mul_grad",
                "sgd",
                "sgd",
                "sgd",
            ],
        )

    def test_multi_checkpoint(self):
        mul_out, b1_out, b2_out, mean_out = self.net()
        self.assertEqual(len(mean_out.block.ops), 4)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            ["mul", "elementwise_add", "elementwise_add", "mean"],
        )
        sgd_optimizer = paddle.optimizer.SGD(learning_rate=1.0)
        recompute_optimizer = paddle.incubate.optimizer.RecomputeOptimizer(
            sgd_optimizer
        )
        recompute_optimizer._set_checkpoints([mul_out, b2_out])
        opts, params_grads = recompute_optimizer.minimize(mean_out)

        self.assertEqual(len(mean_out.block.ops), 13)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            [
                "mul",
                "elementwise_add",
                "elementwise_add",
                "mean",
                "fill_constant",
                "mean_grad",
                "elementwise_add",
                "elementwise_add_grad",
                "elementwise_add_grad",
                "mul_grad",
                "sgd",
                "sgd",
                "sgd",
            ],
        )

    def test_adjacent_checkpoint(self):
        mul_out, b1_out, b2_out, mean_out = self.net()
        self.assertEqual(len(mean_out.block.ops), 4)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            ["mul", "elementwise_add", "elementwise_add", "mean"],
        )
        sgd_optimizer = paddle.optimizer.SGD(learning_rate=1.0)
        recompute_optimizer = paddle.incubate.optimizer.RecomputeOptimizer(
            sgd_optimizer
        )
        recompute_optimizer._set_checkpoints([mul_out, b1_out])
        opts, params_grads = recompute_optimizer.minimize(mean_out)

        self.assertEqual(len(mean_out.block.ops), 12)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            [
                "mul",
                "elementwise_add",
                "elementwise_add",
                "mean",
                "fill_constant",
                "mean_grad",
                "elementwise_add_grad",
                "elementwise_add_grad",
                "mul_grad",
                "sgd",
                "sgd",
                "sgd",
            ],
        )

    def test_out_of_order_checkpoint(self):
        mul_out, b1_out, b2_out, mean_out = self.net()
        self.assertEqual(len(mean_out.block.ops), 4)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            ["mul", "elementwise_add", "elementwise_add", "mean"],
        )
        sgd_optimizer = paddle.optimizer.SGD(learning_rate=1.0)
        recompute_optimizer = paddle.incubate.optimizer.RecomputeOptimizer(
            sgd_optimizer
        )
        recompute_optimizer._set_checkpoints([b2_out, mul_out])
        opts, params_grads = recompute_optimizer.minimize(mean_out)

        self.assertEqual(len(mean_out.block.ops), 13)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            [
                "mul",
                "elementwise_add",
                "elementwise_add",
                "mean",
                "fill_constant",
                "mean_grad",
                "elementwise_add",
                "elementwise_add_grad",
                "elementwise_add_grad",
                "mul_grad",
                "sgd",
                "sgd",
                "sgd",
            ],
        )

    def test_input_as_checkpoints(self):
        mul_x, mul_out, b1_out, b2_out, mean_out = self.net(return_input=True)
        self.assertEqual(len(mean_out.block.ops), 4)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            ["mul", "elementwise_add", "elementwise_add", "mean"],
        )
        sgd_optimizer = paddle.optimizer.SGD(learning_rate=1.0)
        recompute_optimizer = paddle.incubate.optimizer.RecomputeOptimizer(
            sgd_optimizer
        )
        recompute_optimizer._set_checkpoints([mul_x, b2_out])
        opts, params_grads = recompute_optimizer.minimize(mean_out)

        self.assertEqual(len(mean_out.block.ops), 14)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            [
                "mul",
                "elementwise_add",
                "elementwise_add",
                "mean",
                "fill_constant",
                "mean_grad",
                "mul",
                "elementwise_add",
                "elementwise_add_grad",
                "elementwise_add_grad",
                "mul_grad",
                "sgd",
                "sgd",
                "sgd",
            ],
        )

    def test_apply_gradients(self):
        mul_out, b1_out, b2_out, mean_out = self.net()
        sgd_optimizer = paddle.optimizer.SGD(learning_rate=1.0)
        recompute_optimizer = paddle.incubate.optimizer.RecomputeOptimizer(
            sgd_optimizer
        )
        recompute_optimizer._set_checkpoints([b1_out])
        # apply backward
        params_grads = recompute_optimizer.backward(
            mean_out,
            startup_program=None,
            parameter_list=None,
            no_grad_set=None,
        )

        # apply gradient
        program = mean_out.block.program
        with framework.program_guard(program, None):
            optimize_ops = recompute_optimizer.apply_gradients(params_grads)

        self.assertEqual(len(mean_out.block.ops), 13)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            [
                "mul",
                "elementwise_add",
                "elementwise_add",
                "mean",
                "fill_constant",
                "mean_grad",
                "elementwise_add_grad",
                "mul",
                "elementwise_add_grad",
                "mul_grad",
                "sgd",
                "sgd",
                "sgd",
            ],
        )

    def test_load(self):
        mul_out, b1_out, b2_out, mean_out = self.net()
        sgd_optimizer = paddle.optimizer.SGD(learning_rate=1.0)
        recompute_optimizer = paddle.incubate.optimizer.RecomputeOptimizer(
            sgd_optimizer
        )
        recompute_optimizer._set_checkpoints([b1_out])
        try:
            state_dict = {}
            recompute_optimizer.load(state_dict)
        except NotImplementedError as e:
            self.assertEqual(
                "load function is not supported by Recompute Optimizer for now",
                str(e),
            )

    def test_dropout(self):
        """
        If there are dropout layers in the forward nets, we should add a
        seed op
        """
        mul_out, b1_out, b2_out, mean_out = self.net(with_dropout=True)
        self.assertEqual(len(mean_out.block.ops), 5)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            ["mul", "dropout", "elementwise_add", "elementwise_add", "mean"],
        )
        sgd_optimizer = paddle.optimizer.SGD(learning_rate=1.0)
        recompute_optimizer = paddle.incubate.optimizer.RecomputeOptimizer(
            sgd_optimizer
        )
        recompute_optimizer._set_checkpoints([b1_out])
        opts, params_grads = recompute_optimizer.minimize(mean_out)

        self.assertEqual(len(mean_out.block.ops), 17)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            [
                "mul",
                "seed",
                "dropout",
                "elementwise_add",
                "elementwise_add",
                "mean",
                "fill_constant",
                "mean_grad",
                "elementwise_add_grad",
                "mul",
                "dropout",
                "elementwise_add_grad",
                "dropout_grad",
                "mul_grad",
                "sgd",
                "sgd",
                "sgd",
            ],
        )

    def test_dropout_with_determinate_seed(self):
        mul_out, b1_out, b2_out, mean_out = self.net(
            with_dropout=True, with_seed=True
        )
        self.assertEqual(len(mean_out.block.ops), 6)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            [
                "mul",
                "seed",
                "dropout",
                "elementwise_add",
                "elementwise_add",
                "mean",
            ],
        )
        sgd_optimizer = paddle.optimizer.SGD(learning_rate=1.0)
        recompute_optimizer = paddle.incubate.optimizer.RecomputeOptimizer(
            sgd_optimizer
        )
        recompute_optimizer._set_checkpoints([b1_out])
        opts, params_grads = recompute_optimizer.minimize(mean_out)

        self.assertEqual(len(mean_out.block.ops), 17)
        self.assertEqual(
            [op.type for op in mean_out.block.ops],
            [
                "mul",
                "seed",
                "dropout",
                "elementwise_add",
                "elementwise_add",
                "mean",
                "fill_constant",
                "mean_grad",
                "elementwise_add_grad",
                "mul",
                "dropout",
                "elementwise_add_grad",
                "dropout_grad",
                "mul_grad",
                "sgd",
                "sgd",
                "sgd",
            ],
        )

    def test_dropout_with_seed(self):
        """
        when we recompute a dropout op, make sure that the recomputed one
        is the same as the original var.
        """

        def gen_data():
            return {
                "x": np.random.random(size=(100, 3)).astype('float32'),
                "y": np.random.randint(2, size=(100, 1)).astype('int64'),
            }

        def mlp(input_x, input_y):
            drop_res = paddle.nn.functional.dropout(
                input_x, p=0.5, name="dropout_with_seed_cpu"
            )
            prediction = paddle.static.nn.fc(
                x=[drop_res], size=2, activation='softmax'
            )
            drop_res.stop_gradient = False
            cost = paddle.nn.functional.cross_entropy(
                input=prediction,
                label=input_y,
                reduction='none',
                use_softmax=False,
            )
            sum_cost = paddle.mean(cost)
            return drop_res, prediction, sum_cost

        main_program = Program()
        startup_program = Program()
        scope = base.Scope()
        with base.scope_guard(scope):
            with program_guard(main_program, startup_program):
                input_x = paddle.static.data(
                    name="x", shape=[-1, 3], dtype='float32'
                )
                input_y = paddle.static.data(
                    name="y", shape=[-1, 1], dtype='int64'
                )
                drop_res, prediction, cost = mlp(input_x, input_y)
                sgd = paddle.optimizer.Adam(learning_rate=0.01)
                sgd = paddle.incubate.optimizer.RecomputeOptimizer(sgd)
                sgd._set_checkpoints([prediction])
                sgd.minimize(cost)

                place = base.CPUPlace()
                exe = base.Executor(place)
                exe.run(base.default_startup_program())
                feed_data = gen_data()
                drop_vec = exe.run(
                    feed=feed_data,
                    program=base.default_main_program(),
                    fetch_list=[
                        "dropout_with_seed_cpu.tmp_1",
                        "dropout_with_seed_cpu.tmp_1.subprog_0",
                    ],
                )
                self.assertEqual(drop_vec[0].tolist(), drop_vec[1].tolist())


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestRecomputeOptimizerCUDA(unittest.TestCase):
    def test_dropout_with_seed(self):
        """
        when we recompute a dropout op, make sure that the recomputed one
        is the same as the original var.
        """

        def gen_data():
            return {
                "x": np.random.random(size=(100, 3)).astype('float32'),
                "y": np.random.randint(2, size=(100, 1)).astype('int64'),
            }

        def mlp(input_x, input_y):
            drop_res = paddle.nn.functional.dropout(
                input_x, p=0.5, name="dropout_with_seed_gpu"
            )
            prediction = paddle.static.nn.fc(
                x=[drop_res], size=2, activation='softmax'
            )
            drop_res.stop_gradient = False
            cost = paddle.nn.functional.cross_entropy(
                input=prediction,
                label=input_y,
                reduction='none',
                use_softmax=False,
            )
            sum_cost = paddle.mean(cost)
            return drop_res, prediction, sum_cost

        main_program = Program()
        startup_program = Program()
        scope = base.Scope()
        with base.scope_guard(scope):
            with program_guard(main_program, startup_program):
                input_x = paddle.static.data(
                    name="x", shape=[-1, 3], dtype='float32'
                )
                input_y = paddle.static.data(
                    name="y", shape=[-1, 1], dtype='int64'
                )
                drop_res, prediction, cost = mlp(input_x, input_y)
                sgd = paddle.optimizer.Adam(learning_rate=0.01)
                sgd = paddle.incubate.optimizer.RecomputeOptimizer(sgd)
                sgd._set_checkpoints([prediction])
                sgd.minimize(cost)

                place = base.CUDAPlace(0)
                exe = base.Executor(place)
                exe.run(base.default_startup_program())
                feed_data = gen_data()
                drop_vec = exe.run(
                    feed=feed_data,
                    program=base.default_main_program(),
                    fetch_list=[
                        "dropout_with_seed_gpu.tmp_1",
                        "dropout_with_seed_gpu.tmp_1.subprog_0",
                    ],
                )
                self.assertEqual(drop_vec[0].tolist(), drop_vec[1].tolist())


class TestGradientMergeOptimizer(unittest.TestCase):
    def net(self):
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32", shape=[5, 10], name="mul.x"
        )
        mul_y = block.create_var(dtype="float32", shape=[10, 8], name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], name="mul.out"
        )
        b1 = block.create_parameter(dtype="float32", shape=[5, 8], name="b1")
        b1_out = block.create_var(dtype="float32", shape=[5, 8], name="b1_out")
        mean_out = block.create_var(dtype="float32", shape=[1], name="mean.out")
        block.append_op(
            type="mul",
            inputs={"X": mul_x, "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1},
        )
        block.append_op(
            type="elementwise_add",
            inputs={"X": mul_out, "Y": b1},
            outputs={"Out": b1_out},
        )
        block.append_op(
            type="mean", inputs={"X": b1_out}, outputs={"Out": mean_out}
        )
        return mean_out

    def test_program_desc(
        self,
    ):
        cost = self.net()
        main_program = cost.block.program
        init_program = framework.Program()
        self.assertEqual(main_program.num_blocks, 1)
        self.assertEqual(len(cost.block.ops), 3)
        self.assertEqual(
            [op.type for op in cost.block.ops],
            ["mul", "elementwise_add", "mean"],
        )

        opt = paddle.optimizer.SGD(learning_rate=1.0)
        opt = paddle.incubate.optimizer.GradientMergeOptimizer(opt, k_steps=4)
        with framework.program_guard(main_program, init_program):
            ops, params_grads = opt.minimize(cost)

        self.assertEqual(main_program.num_blocks, 2)

        # main block
        self.assertEqual(len(cost.block.ops), 13)
        self.assertEqual(
            [op.type for op in cost.block.ops],
            [
                'mul',
                'elementwise_add',
                'mean',
                'fill_constant',
                'mean_grad',
                'elementwise_add_grad',
                'mul_grad',
                'increment',  # step += 1
                'elementwise_mod',  # step %= k_steps
                'equal',  # cond_var == (step == 0)
                'elementwise_add',
                'elementwise_add',
                'conditional_block',
            ],
        )

        # optimize block
        self.assertEqual(len(main_program.block(1).ops), 6)
        self.assertEqual(
            [op.type for op in main_program.block(1).ops],
            ['scale', 'scale', 'sgd', 'sgd', 'fill_constant', 'fill_constant'],
        )


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
