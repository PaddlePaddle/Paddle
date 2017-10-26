import unittest

import paddle.v2.framework.framework as framework
import paddle.v2.framework.optimizer as optimizer
from paddle.v2.framework.backward import append_backward_ops


class TestOptimizer(unittest.TestCase):
    def test_sgd_optimizer(self):
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32", shape=[5, 10], lod_level=0, name="mul.x")
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
        sgd_optimizer = optimizer.SGDOptimizer(learning_rate=0.01)
        opts = sgd_optimizer.minimize(mul_out)
        self.assertEqual(len(opts), 1)
        sgd_op = opts[0]
        self.assertEqual(sgd_op.type, "sgd")


class TestMomentumOptimizer(unittest.TestCase):
    class MockMomentum(optimizer.MomentumOptimizer):
        def get_accumulators(self):
            return self._accumulators

        def get_velocity_str(self):
            return self._velocity_acc_str

    def test_vanilla_momentum_optimizer(self):
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32", shape=[5, 10], lod_level=0, name="mul.x")
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
        momentum_optimizer = self.MockMomentum(learning_rate=0.01, momentum=0.2)
        params_grads = append_backward_ops(mul_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(momentum_optimizer.get_accumulators()), 0)
        opts = momentum_optimizer.create_optimization_pass(params_grads,
                                                           mul_out)
        self.assertEqual(len(opts), 1)
        sgd_op = opts[0]
        self.assertEqual(sgd_op.type, "momentum")
        self.assertFalse(sgd_op.attr('useNesterov'))

        # Check accumulators
        accumulators = momentum_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 1)
        self.assertTrue(momentum_optimizer.get_velocity_str() in accumulators)
        velocity_acc = accumulators[momentum_optimizer.get_velocity_str()]
        self.assertEqual(len(velocity_acc), 1)
        self.assertTrue(mul_x.name in velocity_acc)

    def test_nesterov_momentum_optimizer(self):
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32", shape=[5, 10], lod_level=0, name="mul.x")
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
        momentum_optimizer = self.MockMomentum(
            learning_rate=0.01, momentum=0.2, use_nesterov=True)
        params_grads = append_backward_ops(mul_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(momentum_optimizer.get_accumulators()), 0)
        opts = momentum_optimizer.create_optimization_pass(params_grads,
                                                           mul_out)
        self.assertEqual(len(opts), 1)
        sgd_op = opts[0]
        self.assertEqual(sgd_op.type, "momentum")
        self.assertTrue(sgd_op.attr('useNesterov'))

        # Check accumulators
        accumulators = momentum_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 1)
        self.assertTrue(momentum_optimizer.get_velocity_str() in accumulators)
        velocity_acc = accumulators[momentum_optimizer.get_velocity_str()]
        self.assertEqual(len(velocity_acc), 1)
        self.assertTrue(mul_x.name in velocity_acc)


class TestAdagradOptimizer(unittest.TestCase):
    class MockAdagrad(optimizer.AdagradOptimizer):
        def get_accumulators(self):
            return self._accumulators

        def get_moment_str(self):
            return self._moment_acc_str

    def test_adagrad_optimizer(self):
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32", shape=[5, 10], lod_level=0, name="mul.x")
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
        adagrad_optimizer = self.MockAdagrad(learning_rate=0.01, epsilon=1.0e-6)
        params_grads = append_backward_ops(mul_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(adagrad_optimizer.get_accumulators()), 0)
        opts = adagrad_optimizer.create_optimization_pass(params_grads, mul_out)
        self.assertEqual(len(opts), 1)
        adagrad_op = opts[0]
        self.assertEqual(adagrad_op.type, "adagrad")

        # check accumulators
        accumulators = adagrad_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 1)
        self.assertTrue(adagrad_optimizer.get_moment_str() in accumulators)
        moment_acc = accumulators[adagrad_optimizer.get_moment_str()]
        self.assertEqual(len(moment_acc), 1)
        self.assertTrue(mul_x.name in moment_acc)


class TestAdamOptimizer(unittest.TestCase):
    class MockAdam(optimizer.AdamOptimizer):
        def get_accumulators(self):
            return self._accumulators

        def get_moment1_str(self):
            return self._moment1_acc_str

        def get_moment2_str(self):
            return self._moment2_acc_str

    def test_adam_optimizer(self):
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32", shape=[5, 10], lod_level=0, name="mul.x")
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
        adam_optimizer = self.MockAdam(
            learning_rate=0.01, beta1=0.9, beta2=0.999)
        params_grads = append_backward_ops(mul_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(adam_optimizer.get_accumulators()), 0)
        opts = adam_optimizer.create_optimization_pass(params_grads, mul_out)
        self.assertEqual(len(opts), 3)
        adam_op = opts[0]
        self.assertEqual(adam_op.type, "adam")

        # Check accumulators
        accumulators = adam_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 2)
        self.assertTrue(adam_optimizer.get_moment1_str() in accumulators)
        self.assertTrue(adam_optimizer.get_moment2_str() in accumulators)
        moment1_acc = accumulators[adam_optimizer.get_moment1_str()]
        moment2_acc = accumulators[adam_optimizer.get_moment2_str()]
        self.assertEqual(len(moment1_acc), 1)
        self.assertEqual(len(moment2_acc), 1)
        self.assertTrue(mul_x.name in moment1_acc)
        self.assertTrue(mul_x.name in moment2_acc)


class TestAdamaxOptimizer(unittest.TestCase):
    class MockAdamax(optimizer.AdamaxOptimizer):
        def get_accumulators(self):
            return self._accumulators

        def get_moment_str(self):
            return self._moment_acc_str

        def get_inf_norm_str(self):
            return self._inf_norm_acc_str

    def test_adamax_optimizer(self):
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32", shape=[5, 10], lod_level=0, name="mul.x")
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
        adamax_optimizer = self.MockAdamax(
            learning_rate=0.01, beta1=0.9, beta2=0.999)
        params_grads = append_backward_ops(mul_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(adamax_optimizer.get_accumulators()), 0)
        opts = adamax_optimizer.create_optimization_pass(params_grads, mul_out)
        self.assertEqual(len(opts), 2)
        adam_op = opts[0]
        self.assertEqual(adam_op.type, "adamax")

        # Check accumulators
        accumulators = adamax_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 2)
        self.assertTrue(adamax_optimizer.get_moment_str() in accumulators)
        self.assertTrue(adamax_optimizer.get_inf_norm_str() in accumulators)
        moment_acc = accumulators[adamax_optimizer.get_moment_str()]
        inf_norm_acc = accumulators[adamax_optimizer.get_inf_norm_str()]
        self.assertEqual(len(moment_acc), 1)
        self.assertEqual(len(inf_norm_acc), 1)
        self.assertTrue(mul_x.name in moment_acc)
        self.assertTrue(mul_x.name in inf_norm_acc)


if __name__ == '__main__':
    unittest.main()
