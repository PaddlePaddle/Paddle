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

import paddle.fluid.framework as framework
import paddle.fluid.optimizer as optimizer
import paddle.compat as cpt
from paddle.fluid.backward import append_backward


class TestOptimizer(unittest.TestCase):
    def test_sgd_optimizer(self):
        def check_sgd_optimizer(optimizer_attr):
            init_program = framework.Program()
            program = framework.Program()
            block = program.global_block()
            mul_x = block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="mul.x",
                optimize_attr=optimizer_attr)
            mul_y = block.create_var(
                dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
            mul_out = block.create_var(
                dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
            mean_out = block.create_var(
                dtype="float32", shape=[1], lod_level=0, name="mean.out")
            block.append_op(
                type="mul",
                inputs={"X": mul_x,
                        "Y": mul_y},
                outputs={"Out": mul_out},
                attrs={"x_num_col_dims": 1})
            block.append_op(
                type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out})
            sgd_optimizer = optimizer.SGDOptimizer(learning_rate=0.01)
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
                lod_level=0,
                name="mul.x",
                optimize_attr=optimizer_attr)
            mul_y = block.create_var(
                dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
            mul_out = block.create_var(
                dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
            mean_out = block.create_var(
                dtype="float32", shape=[1], lod_level=0, name="mean.out")
            block.append_op(
                type="mul",
                inputs={"X": mul_x,
                        "Y": mul_y},
                outputs={"Out": mul_out},
                attrs={"x_num_col_dims": 1})
            block.append_op(
                type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out})
            sgd_optimizer = optimizer.SGDOptimizer(learning_rate=0.01)
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
    class MockMomentum(optimizer.MomentumOptimizer):
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
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1})
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
        learning_rate = 0.01
        momentum_optimizer = self.MockMomentum(
            learning_rate=learning_rate, momentum=0.2)
        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out")
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out})
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
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)
        self.assertEqual(init_ops[1].type, "fill_constant")
        self.assertAlmostEqual(init_ops[1].attr('value'), 0.0)

    def test_nesterov_momentum_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1})
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
        learning_rate = 0.01
        momentum_optimizer = self.MockMomentum(
            learning_rate=learning_rate, momentum=0.2, use_nesterov=True)
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
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)
        self.assertEqual(init_ops[1].type, "fill_constant")
        self.assertAlmostEqual(init_ops[1].attr('value'), 0.0)


class TestAdagradOptimizer(unittest.TestCase):
    class MockAdagrad(optimizer.AdagradOptimizer):
        def get_accumulators(self):
            return self._accumulators

        def get_moment_str(self):
            return self._moment_acc_str

    def test_adagrad_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1})
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
        learning_rate = 0.01
        adagrad_optimizer = self.MockAdagrad(
            learning_rate=learning_rate, epsilon=1.0e-6)
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(adagrad_optimizer.get_accumulators()), 0)
        with framework.program_guard(program, init_program):
            opts = adagrad_optimizer.apply_gradients(params_grads)
        self.assertEqual(len(opts), 2)
        self.assertEqual([op.type for op in opts], ["scale", "adagrad"])

        # Check accumulators
        accumulators = adagrad_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 1)
        self.assertTrue(adagrad_optimizer.get_moment_str() in accumulators)
        moment_acc = accumulators[adagrad_optimizer.get_moment_str()]
        self.assertEqual(len(moment_acc), 1)
        self.assertTrue(mul_x.name in moment_acc)

        # Check init_program
        init_ops = init_program.global_block().ops
        self.assertEqual(len(init_ops), 3)
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)
        self.assertEqual(init_ops[1].type, "fill_constant")
        self.assertAlmostEqual(init_ops[1].attr('value'), 0.0)


class TestAdamOptimizer(unittest.TestCase):
    class MockAdam(optimizer.AdamOptimizer):
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
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1})
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
        learning_rate = 0.01
        adam_optimizer = self.MockAdam(
            learning_rate=learning_rate, beta1=0.9, beta2=0.999)
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(adam_optimizer.get_accumulators()), 0)
        with framework.program_guard(program, init_program):
            opts = adam_optimizer.apply_gradients(params_grads)
        self.assertEqual(len(opts), 4)
        self.assertEqual([op.type for op in opts],
                         ["scale", "adam", "scale", "scale"])

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
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)


class TestAdamaxOptimizer(unittest.TestCase):
    class MockAdamax(optimizer.AdamaxOptimizer):
        def get_accumulators(self):
            return self._accumulators

        def get_moment_str(self):
            return self._moment_acc_str

        def get_inf_norm_str(self):
            return self._inf_norm_acc_str

    def test_adamax_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1})
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
        learning_rate = 0.01
        adamax_optimizer = self.MockAdamax(
            learning_rate=learning_rate, beta1=0.9, beta2=0.999)
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(adamax_optimizer.get_accumulators()), 0)
        with framework.program_guard(program, init_program):
            opts = adamax_optimizer.apply_gradients(params_grads)
        self.assertEqual(len(opts), 3)
        self.assertEqual([op.type for op in opts], ["scale", "adamax", "scale"])

        # Check accumulators
        accumulators = adamax_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 3)
        self.assertTrue(adamax_optimizer.get_moment_str() in accumulators)
        self.assertTrue(adamax_optimizer.get_inf_norm_str() in accumulators)
        moment_acc = accumulators[adamax_optimizer.get_moment_str()]
        inf_norm_acc = accumulators[adamax_optimizer.get_inf_norm_str()]
        self.assertEqual(len(moment_acc), 1)
        self.assertEqual(len(inf_norm_acc), 1)
        self.assertTrue(mul_x.name in moment_acc)
        self.assertTrue(mul_x.name in inf_norm_acc)

        # Check init_program
        init_ops = init_program.global_block().ops
        self.assertEqual(len(init_ops), 4)
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)


class TestDpsgdOptimizer(unittest.TestCase):
    def test_dpsgd_optimizer(self):
        def check_dpsgd_optimizer(optimizer_attr):
            init_program = framework.Program()
            program = framework.Program()
            block = program.global_block()
            mul_x = block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="mul.x",
                optimize_attr=optimizer_attr)
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
            dpsgd_optimizer = optimizer.DpsgdOptimizer(
                learning_rate=0.01, clip=100.0, batch_size=16.0, sigma=0.0)
            opts, _ = dpsgd_optimizer.minimize(mean_out, init_program)
            return opts

        opts = check_dpsgd_optimizer({
            'learning_rate': 1.1,
            'clip': 100.0,
            'batch_size': 16.0,
            'sigma': 4.0
        })
        self.assertEqual(len(opts), 2)
        self.assertEqual([op.type for op in opts], ["scale", "dpsgd"])


class TestDecayedAdagradOptimizer(unittest.TestCase):
    class MockDecayedAdagrad(optimizer.DecayedAdagradOptimizer):
        def get_accumulators(self):
            return self._accumulators

        def get_moment_str(self):
            return self._moment_acc_str

    def test_decayed_adagrad_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1})
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
        learning_rate = 0.01
        decayed_adagrad_optimizer = self.MockDecayedAdagrad(
            learning_rate=learning_rate, decay=0.95, epsilon=1.0e-6)
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(decayed_adagrad_optimizer.get_accumulators()), 0)
        with framework.program_guard(program, init_program):
            opts = decayed_adagrad_optimizer.apply_gradients(params_grads)
        self.assertEqual(len(opts), 2)
        self.assertEqual([op.type for op in opts], ["scale", "decayed_adagrad"])

        # Check accumulators
        accumulators = decayed_adagrad_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 1)
        self.assertTrue(
            decayed_adagrad_optimizer.get_moment_str() in accumulators)
        moment_acc = accumulators[decayed_adagrad_optimizer.get_moment_str()]
        self.assertEqual(len(moment_acc), 1)
        self.assertTrue(mul_x.name in moment_acc)

        # Check init_program
        init_ops = init_program.global_block().ops
        self.assertEqual(len(init_ops), 2)
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)
        self.assertEqual(init_ops[1].type, "fill_constant")
        self.assertAlmostEqual(init_ops[1].attr('value'), 0.0)


class TestFtrlOptimizer(unittest.TestCase):
    class MockFtrl(optimizer.FtrlOptimizer):
        def get_accumulators(self):
            return self._accumulators

        def get_squared_str(self):
            return self._squared_acc_str

        def get_linear_str(self):
            return self._linear_acc_str

    def test_ftrl_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1})
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
        learning_rate = 0.01
        ftrl_optimizer = self.MockFtrl(
            learning_rate=learning_rate, l1=0.0, l2=0.0, lr_power=-0.5)
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(ftrl_optimizer.get_accumulators()), 0)
        with framework.program_guard(program, init_program):
            opts = ftrl_optimizer.apply_gradients(params_grads)
        self.assertEqual(len(opts), 2)
        self.assertEqual([op.type for op in opts], ["scale", "ftrl"])

        # Check accumulators
        accumulators = ftrl_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 2)
        self.assertTrue(ftrl_optimizer.get_squared_str() in accumulators)
        self.assertTrue(ftrl_optimizer.get_linear_str() in accumulators)
        squared_acc = accumulators[ftrl_optimizer.get_squared_str()]
        linear_acc = accumulators[ftrl_optimizer.get_linear_str()]
        self.assertEqual(len(squared_acc), 1)
        self.assertEqual(len(linear_acc), 1)
        self.assertTrue(mul_x.name in squared_acc)
        self.assertTrue(mul_x.name in linear_acc)

        # Check init_program
        init_ops = init_program.global_block().ops
        self.assertEqual(len(init_ops), 3)
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)


class TestLookaheadOptimizer(unittest.TestCase):
    def test_lookahead_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        init_block = init_program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1})
        init_mul_x = init_block.create_parameter(
            dtype="float32", shape=[5, 10], lod_level=0, name="mul.x")
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out")

        block.append_op(
            type="mul",
            inputs={"X": mul_x,
                    "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1})
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out})

        sgd = optimizer.SGD(learning_rate=0.01)
        lookahead = optimizer.LookaheadOptimizer(sgd, alpha=0.5, k=5)
        with framework.program_guard(program, init_program):
            opts, _ = lookahead.minimize(mean_out)
        self.assertEqual(len(opts), 2)
        self.assertEqual([op.type for op in opts], ["scale", "sgd"])


class TestRecomputeOptimizer(unittest.TestCase):
    def net(self):
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32", shape=[5, 10], lod_level=0, name="mul.x")
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
        b1 = block.create_parameter(
            dtype="float32", shape=[5, 8], lod_level=0, name="b1")
        b1_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="b1_out")
        b2 = block.create_parameter(
            dtype="float32", shape=[5, 8], lod_level=0, name="b2")
        b2_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="b2_out")
        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out")
        block.append_op(
            type="mul",
            inputs={"X": mul_x,
                    "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1})
        block.append_op(
            type="elementwise_add",
            inputs={"X": mul_out,
                    "Y": b1},
            outputs={"Out": b1_out})
        block.append_op(
            type="elementwise_add",
            inputs={"X": b1_out,
                    "Y": b2},
            outputs={"Out": b2_out})
        block.append_op(
            type="mean", inputs={"X": b2_out}, outputs={"Out": mean_out})

        return mul_out, b1_out, b2_out, mean_out

    def test_no_checkpoint(self):
        mul_out, b1_out, b2_out, mean_out = self.net()
        self.assertEqual(len(mean_out.block.ops), 4)
        self.assertEqual([op.type for op in mean_out.block.ops],
                         ["mul", "elementwise_add", "elementwise_add", "mean"])
        sgd_optimizer = optimizer.SGD(learning_rate=1.0)
        recompute_optimizer = optimizer.RecomputeOptimizer(sgd_optimizer)
        recompute_optimizer._set_checkpoints([])
        opts, params_grads = recompute_optimizer.minimize(mean_out)

        self.assertEqual(len(mean_out.block.ops), 12)
        self.assertEqual([op.type for op in mean_out.block.ops], [
            "mul", "elementwise_add", "elementwise_add", "mean",
            "fill_constant", "mean_grad", "elementwise_add_grad",
            "elementwise_add_grad", "mul_grad", "sgd", "sgd", "sgd"
        ])

    def test_one_checkpoint(self):
        mul_out, b1_out, b2_out, mean_out = self.net()
        self.assertEqual(len(mean_out.block.ops), 4)
        self.assertEqual([op.type for op in mean_out.block.ops],
                         ["mul", "elementwise_add", "elementwise_add", "mean"])
        sgd_optimizer = optimizer.SGD(learning_rate=1.0)
        recompute_optimizer = optimizer.RecomputeOptimizer(sgd_optimizer)
        recompute_optimizer._set_checkpoints([b1_out])
        opts, params_grads = recompute_optimizer.minimize(mean_out)

        self.assertEqual(len(mean_out.block.ops), 13)
        self.assertEqual([op.type for op in mean_out.block.ops], [
            "mul", "elementwise_add", "elementwise_add", "mean",
            "fill_constant", "mean_grad", "elementwise_add_grad", "mul",
            "elementwise_add_grad", "mul_grad", "sgd", "sgd", "sgd"
        ])

    def test_multi_checkpoint(self):
        mul_out, b1_out, b2_out, mean_out = self.net()
        self.assertEqual(len(mean_out.block.ops), 4)
        self.assertEqual([op.type for op in mean_out.block.ops],
                         ["mul", "elementwise_add", "elementwise_add", "mean"])
        sgd_optimizer = optimizer.SGD(learning_rate=1.0)
        recompute_optimizer = optimizer.RecomputeOptimizer(sgd_optimizer)
        recompute_optimizer._set_checkpoints([mul_out, b2_out])
        opts, params_grads = recompute_optimizer.minimize(mean_out)

        self.assertEqual(len(mean_out.block.ops), 13)
        self.assertEqual([op.type for op in mean_out.block.ops], [
            "mul", "elementwise_add", "elementwise_add", "mean",
            "fill_constant", "mean_grad", "elementwise_add",
            "elementwise_add_grad", "elementwise_add_grad", "mul_grad", "sgd",
            "sgd", "sgd"
        ])

    def test_adjacent_checkpoint(self):
        mul_out, b1_out, b2_out, mean_out = self.net()
        self.assertEqual(len(mean_out.block.ops), 4)
        self.assertEqual([op.type for op in mean_out.block.ops],
                         ["mul", "elementwise_add", "elementwise_add", "mean"])
        sgd_optimizer = optimizer.SGD(learning_rate=1.0)
        recompute_optimizer = optimizer.RecomputeOptimizer(sgd_optimizer)
        recompute_optimizer._set_checkpoints([mul_out, b1_out])
        opts, params_grads = recompute_optimizer.minimize(mean_out)

        self.assertEqual(len(mean_out.block.ops), 12)
        self.assertEqual([op.type for op in mean_out.block.ops], [
            "mul", "elementwise_add", "elementwise_add", "mean",
            "fill_constant", "mean_grad", "elementwise_add_grad",
            "elementwise_add_grad", "mul_grad", "sgd", "sgd", "sgd"
        ])

    def test_apply_gradients(self):
        mul_out, b1_out, b2_out, mean_out = self.net()
        sgd_optimizer = optimizer.SGD(learning_rate=1.0)
        recompute_optimizer = optimizer.RecomputeOptimizer(sgd_optimizer)
        recompute_optimizer._set_checkpoints([b1_out])
        # apply backward
        params_grads = recompute_optimizer.backward(
            mean_out,
            startup_program=None,
            parameter_list=None,
            no_grad_set=None,
            checkpoints=[b1_out])

        # apply gradient
        program = mean_out.block.program
        with framework.program_guard(program, None):
            optimize_ops = recompute_optimizer.apply_gradients(params_grads)

        self.assertEqual(len(mean_out.block.ops), 13)
        self.assertEqual([op.type for op in mean_out.block.ops], [
            "mul", "elementwise_add", "elementwise_add", "mean",
            "fill_constant", "mean_grad", "elementwise_add_grad", "mul",
            "elementwise_add_grad", "mul_grad", "sgd", "sgd", "sgd"
        ])

    def test_load(self):
        mul_out, b1_out, b2_out, mean_out = self.net()
        sgd_optimizer = optimizer.SGD(learning_rate=1.0)
        recompute_optimizer = optimizer.RecomputeOptimizer(sgd_optimizer)
        recompute_optimizer._set_checkpoints([b1_out])
        try:
            stat_dict = {}
            recompute_optimizer.load(stat_dict)
        except NotImplementedError as e:
            self.assertEqual(
                "load function is not supported by Recompute Optimizer for now",
                cpt.get_exception_message(e))


if __name__ == '__main__':
    unittest.main()
