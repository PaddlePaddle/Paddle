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
from paddle.base import core
from paddle.base.dygraph.base import switch_to_static_graph


class LAMBOptimizer(paddle.optimizer.Lamb):
    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, (base.framework.Block, paddle.pir.Block))
        block.program._use_lamb = True

        m = moment1 = self._get_accumulator(
            self._moment1_acc_str, param_and_grad[0]
        )
        v = self._get_accumulator(self._moment2_acc_str, param_and_grad[0])
        beta_1_pow_acc = self._get_accumulator(
            self._beta1_pow_acc_str, param_and_grad[0]
        )
        beta_2_pow_acc = self._get_accumulator(
            self._beta2_pow_acc_str, param_and_grad[0]
        )

        beta_1 = paddle.tensor.fill_constant(
            dtype='float32', shape=[1], value=self._beta1, name='lamb_beta_1'
        )
        beta_2 = paddle.tensor.fill_constant(
            dtype='float32', shape=[1], value=self._beta2, name='lamb_beta_2'
        )
        epsilon = paddle.tensor.fill_constant(
            dtype='float32', shape=[1], value=self._epsilon, name='epsilon'
        )

        one = paddle.ones(shape=[1]).astype('float32')
        zero = paddle.zeros(shape=[1]).astype('float32')

        next_m = paddle.multiply(m, beta_1) + paddle.multiply(
            param_and_grad[1], one - beta_1
        )
        next_v = paddle.multiply(v, beta_2) + paddle.multiply(
            paddle.pow(param_and_grad[1], 2), one - beta_2
        )

        beta1_correction = one - beta_1_pow_acc
        beta2_correction = one - beta_2_pow_acc

        next_m_unbiased = next_m / beta1_correction
        next_v_unbiased = next_v / beta2_correction

        update = next_m_unbiased / (paddle.sqrt(next_v_unbiased) + epsilon)

        if (
            self._exclude_from_weight_decay_fn is not None
            and self._exclude_from_weight_decay_fn(param_and_grad[0])
        ):
            self._lamb_weight_decay = 0.0
        update += self._lamb_weight_decay * param_and_grad[0]

        w_norm = paddle.norm(param_and_grad[0], p=2)
        g_norm = paddle.norm(update, p=2)

        learning_rate = self._create_param_lr(param_and_grad)

        ratio = paddle.where(
            paddle.greater_than(w_norm, zero),
            paddle.where(
                paddle.greater_than(g_norm, zero), (w_norm / g_norm), one
            ),
            one,
        )
        update_with_lr = ratio * learning_rate * update
        next_param = param_and_grad[0] - update_with_lr

        beta_1_pow_acc *= beta_1
        beta_2_pow_acc *= beta_2

        paddle.assign(next_m, m)
        paddle.assign(next_v, v)
        paddle.assign(next_param, param_and_grad[0])


class TestLambOpV2(unittest.TestCase):
    def test_lamb_op(self):
        shape = [2, 4, 8, 8]
        data = paddle.to_tensor(np.random.random(size=shape).astype("float32"))
        conv = paddle.nn.Conv2D(4, 6, (3, 3))
        data = conv(data)
        loss = paddle.mean(data)
        opt = paddle.optimizer.Lamb(
            learning_rate=1e-5, epsilon=1e-8, parameters=conv.parameters()
        )
        loss.backward()
        opt.minimize(loss)

        assert loss.numpy() is not None


class TestLambOpWithCombinedOp(unittest.TestCase):

    def test_lamb_op_with_multi_steps(self):
        paddle.enable_static()

        def _build_static_model(main, startup, seed=100):
            with base.program_guard(main, startup):
                paddle.seed(seed)
                x = paddle.static.data(
                    name='X', shape=[-1, 13], dtype='float32'
                )
                y = paddle.static.data(name='Y', shape=[-1, 1], dtype='float32')
                linear = paddle.nn.Linear(
                    in_features=x.shape[-1], out_features=1
                )
                prediction = linear(x)
                loss = paddle.nn.functional.square_error_cost(
                    input=prediction, label=y
                )
                avg_loss = paddle.mean(loss)
            return avg_loss

        place = base.CPUPlace()
        num_steps = 10

        for i in range(num_steps):
            feed_x = np.random.random(size=(10, 13)).astype('float32')
            feed_y = np.random.random(size=(10, 1)).astype('float32')

            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with base.program_guard(main_program, startup_program):
                avg_loss = _build_static_model(main_program, startup_program)
                lamb_kernel = paddle.optimizer.Lamb(learning_rate=0.2)
                lamb_kernel.minimize(avg_loss)

            executor = base.Executor(place)
            executor.run(startup_program)
            output = executor.run(
                program=main_program,
                feed={'X': feed_x, 'Y': feed_y},
                fetch_list=[avg_loss],
            )

            main = paddle.static.Program()
            startup = paddle.static.Program()
            with base.program_guard(main, startup):
                loss = _build_static_model(main, startup)
                lamb = LAMBOptimizer(learning_rate=0.2)
                lamb.minimize(loss)

            exe = base.Executor(place)
            exe.run(startup)
            out = exe.run(
                program=main,
                feed={'X': feed_x, 'Y': feed_y},
                fetch_list=[loss],
            )

            np.testing.assert_allclose(out, output, rtol=1e-05)


class TestLambOpV2Group(TestLambOpV2):
    def test_lamb_op(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.Lamb(
            learning_rate=0.01,
            parameters=[
                {'params': linear_1.parameters()},
                {
                    'params': linear_2.parameters(),
                    'lamb_weight_decay': 0.001,
                    'beta1': 0.9,
                    'beta2': 0.99,
                },
            ],
            lamb_weight_decay=0.01,
        )
        out = linear_1(a)
        out = linear_2(out)
        out.backward()
        adam.step()
        adam.clear_gradients()


class TestLambOpMultiPrecision(unittest.TestCase):
    def check_main(self, x_np, place, multi_precision=False, seed=10, n=10):
        with paddle.pir_utils.OldIrGuard():
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                paddle.seed(seed)
                with paddle.static.amp.fp16_guard():
                    x = paddle.static.data(
                        name='x', shape=[None, 10], dtype='float32'
                    )
                    linear = paddle.nn.Linear(10, 2)
                    hidden = linear(x)
                    loss = paddle.mean(hidden)

                original_optimizer = paddle.optimizer.Lamb(learning_rate=1e-3)
                original_optimizer._multi_precision = multi_precision
                if multi_precision:
                    optimizer = paddle.static.amp.decorate(
                        original_optimizer,
                        use_pure_fp16=True,
                        use_fp16_guard=True,
                    )
                else:
                    optimizer = original_optimizer
                optimizer.minimize(loss)

            weight, bias = linear.weight, linear.bias
            exe = paddle.static.Executor(place)
            scope = paddle.static.Scope()
            if x.dtype in (core.VarDesc.VarType.FP16, core.DataType.FLOAT16):
                x_np = x_np.astype(np.float16)

            def get_parameter(var):
                name = var if isinstance(var, (str, bytes)) else var.name
                params = original_optimizer._get_parameter(name, scope)
                assert isinstance(params, (list, tuple))
                params = list(params)
                assert len(params) == 2
                if multi_precision:
                    params[0] = np.array(params[0])
                    params[1] = np.array(params[1])
                    np.testing.assert_array_equal(
                        params[0], params[1].astype(np.float16)
                    )
                    return params[0].astype(np.float32)
                else:
                    self.assertIsNotNone(params[0])
                    self.assertIsNone(params[1])
                    params[0] = np.array(params[0])
                    return params[0]

            with paddle.static.scope_guard(scope):
                exe.run(startup_prog)
                if multi_precision:
                    optimizer.amp_init(place)

                weight_np, bias_np = None, None
                for i in range(n):
                    feed_dict = {'x': x_np}
                    weight_np, bias_np = exe.run(
                        main_prog, feed=feed_dict, fetch_list=[weight, bias]
                    )
                    weight_np = weight_np.astype('float32')
                    bias_np = bias_np.astype('float32')
                    np.testing.assert_array_equal(
                        weight_np, get_parameter(weight)
                    )
                    np.testing.assert_array_equal(bias_np, get_parameter(bias))
                return weight_np, bias_np

    def check_amp_in_pir(
        self, x_np, place, multi_precision=True, seed=10, n=10
    ):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                paddle.seed(seed)

                x = paddle.static.data(
                    name='x', shape=[None, 10], dtype='float32'
                )
                linear = paddle.nn.Linear(10, 2)
                original_optimizer = paddle.optimizer.Lamb(
                    learning_rate=0.001, parameters=linear.parameters()
                )

                linear, optimizer = paddle.amp.decorate(
                    models=linear,
                    optimizers=original_optimizer,
                    level='O2',
                )

                with paddle.amp.auto_cast(
                    level='O2', dtype='float16', use_promote=True
                ):
                    out = linear(x)
                    loss = paddle.mean(out)
                optimizer.minimize(loss)

            weight, bias = linear.weight, linear.bias
            exe = paddle.static.Executor(place)

            def get_parameter(var):
                name = var if isinstance(var, (str, bytes)) else var.name
                params = original_optimizer._get_parameter(name)
                assert isinstance(params, (list, tuple))
                params = list(params)
                assert len(params) == 2
                if multi_precision:
                    params[0] = np.array(params[0])
                    params[1] = np.array(params[1])
                    np.testing.assert_array_equal(
                        params[0], params[1].astype(np.float16)
                    )
                    return params[0].astype(np.float32)
                else:
                    self.assertIsNotNone(params[0])
                    self.assertIsNone(params[1])
                    params[0] = np.array(params[0])
                    return params[0]

            exe.run(startup_prog)
            if multi_precision:
                optimizer.amp_init(place)

            weight_np, bias_np = None, None
            for i in range(n):
                feed_dict = {'x': x_np}
                weight_np, bias_np = exe.run(
                    main_prog, feed=feed_dict, fetch_list=[weight, bias]
                )
                weight_np = weight_np.astype('float32')
                bias_np = bias_np.astype('float32')
                np.testing.assert_array_equal(weight_np, get_parameter(weight))
                np.testing.assert_array_equal(bias_np, get_parameter(bias))

            return weight_np, bias_np

    @switch_to_static_graph
    def test_main(self):
        if not paddle.is_compiled_with_cuda():
            return

        place = paddle.CUDAPlace(0)
        x_np = np.random.random(size=[5, 10]).astype('float32')
        weight_1, bias_1 = self.check_main(x_np, place, multi_precision=False)
        weight_2, bias_2 = self.check_main(x_np, place, multi_precision=True)
        weight_3, bias_3 = self.check_amp_in_pir(x_np, place)
        self.assertTrue(np.all(np.abs(weight_1 - weight_2) < 1e-3))
        self.assertTrue(np.all(np.abs(bias_1 - bias_2) < 1e-7))
        self.assertTrue(np.all(np.abs(weight_1 - weight_3) < 1e-3))
        self.assertTrue(np.all(np.abs(bias_1 - bias_3) < 1e-7))


if __name__ == "__main__":
    unittest.main()
