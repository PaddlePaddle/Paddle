# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import paddle
from paddle import base
from paddle.framework import core


def nadam_step(inputs, attributes):
    param = inputs['param']
    grad = inputs['grad']
    lr = inputs['learning_rate']

    # accumulators
    momentum_decay_pow = inputs['momentum_decay_pow']
    beta2_pow = inputs['beta2_pow']
    mu_product = inputs['mu_product']
    moment1 = inputs['moment1']
    moment2 = inputs['moment2']

    # attrs
    epsilon = attributes['epsilon']
    beta1 = attributes['beta1']
    beta2 = attributes['beta2']
    momentum_decay = attributes['momentum_decay']
    momentum_decay_base = attributes['momentum_decay_base']

    momentum_decay_pow *= momentum_decay_base
    beta2_pow *= beta2

    mu_t = beta1 * (1.0 - 0.5 * (momentum_decay_pow**momentum_decay))
    mu_t_1 = beta1 * (
        1.0
        - 0.5
        * (momentum_decay_pow**momentum_decay)
        * (momentum_decay_base**momentum_decay)
    )

    mu_product *= mu_t
    mu_product_t_1 = mu_product * mu_t_1

    moment1 = beta1 * moment1 + (1.0 - beta1) * grad
    moment2 = beta2 * moment2 + (1.0 - beta2) * grad * grad

    moment1_hat = mu_t_1 * moment1 / (1.0 - mu_product_t_1) + (
        1.0 - mu_t
    ) * grad / (1.0 - mu_product)
    moment2_hat = moment2 / (1.0 - beta2_pow)

    param = param - lr * moment1_hat / (np.sqrt(moment2_hat) + epsilon)

    # get accumulators
    return param, momentum_decay_pow, beta2_pow, mu_product, moment1, moment2


def nadam_wrapper(
    param,
    grad,
    lr,
    momentum_decay_pow,
    beta2_pow,
    mu_product,
    moment1,
    moment2,
    master_param=None,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    momentum_decay=0.004,
    momentum_decay_base=0.96,
    multi_precision=False,
):
    _, _, _, _, _, _, _ = paddle._C_ops.nadam_(
        param,
        grad,
        lr,
        momentum_decay_pow,
        beta2_pow,
        mu_product,
        moment1,
        moment2,
        master_param,
        beta1,
        beta2,
        epsilon,
        momentum_decay,
        momentum_decay_base,
        multi_precision,
    )


class TestNAdamOp(OpTest):
    def setUp(self):
        '''Test NAdam Op with supplied attributes'''
        np.random.seed(2024)

        self.op_type = "nadam"
        self.python_api = nadam_wrapper
        self.python_out_sig = ['out']

        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        learning_rate = np.array(0.003).astype("float32")

        beta1 = 0.78
        beta2 = 0.915
        epsilon = 1e-8
        momentum_decay = 0.004
        momentum_decay_base = 0.96

        # accumulators
        momentum_decay_pow = np.array(momentum_decay_base**3).astype(
            "float32"
        )
        # use beta1 to fake mu_product
        mu_product = np.array(beta1**3).astype("float32")
        beta2_pow = np.array(beta2**3).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

        self.inputs = {
            "param": param,
            "grad": grad,
            "momentum_decay_pow": momentum_decay_pow,
            "beta2_pow": beta2_pow,
            "mu_product": mu_product,
            "moment1": moment1,
            "moment2": moment2,
            "learning_rate": learning_rate,
        }

        self.attrs = {
            "epsilon": epsilon,
            "beta1": beta1,
            "beta2": beta2,
            "momentum_decay": momentum_decay,
            "momentum_decay_base": momentum_decay_base,
        }

        (
            param_out,
            momentum_decay_pow_out,
            beta2_pow_out,
            mu_product_out,
            moment1_out,
            moment2_out,
        ) = nadam_step(self.inputs, self.attrs)

        self.outputs = {
            "param_out": param_out,
            "momentum_decay_pow_out": momentum_decay_pow_out,
            "beta2_pow_out": beta2_pow_out,
            "mu_product_out": mu_product_out,
            "moment1_out": moment1_out,
            "moment2_out": moment2_out,
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestNAdamOpGPU(TestNAdamOp):
    def test_check_output(self):
        self.check_output_with_place(core.CUDAPlace(0), check_pir=True)


class TestNAdamAPI(unittest.TestCase):
    def test_nadam_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        nadam = paddle.optimizer.NAdam(
            learning_rate=0.01,
            parameters=linear.parameters(),
            weight_decay=0.01,
        )

        for _ in range(2):
            out = linear(a)
            out.backward()
            nadam.step()
            nadam.clear_gradients()

    def test_nadam_apply_gradients(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        nadam = paddle.optimizer.NAdam(
            learning_rate=0.01,
            parameters=linear.parameters(),
            weight_decay=0.01,
        )

        for _ in range(2):
            out = linear(a)
            loss = paddle.mean(out)
            param_grads = nadam.backward(loss)
            nadam.apply_gradients(param_grads)
            nadam.clear_gradients()

    def test_nadam_static(self):
        paddle.enable_static()
        place = base.CPUPlace()
        shape = [2, 3, 8, 8]
        exe = base.Executor(place)
        train_prog = base.Program()
        startup = base.Program()
        with base.program_guard(train_prog, startup):
            with base.unique_name.guard():
                data = paddle.static.data(name="data", shape=shape)
                conv = paddle.static.nn.conv2d(data, 8, 3)
                loss = paddle.mean(conv)

                beta1 = 0.85
                beta2 = 0.95
                opt = paddle.optimizer.NAdam(
                    learning_rate=1e-5,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=0.01,
                    epsilon=1e-8,
                )
                opt.minimize(loss)

        exe.run(startup)
        data_np = np.random.random(shape).astype('float32')
        rets = exe.run(train_prog, feed={"data": data_np}, fetch_list=[loss])
        assert rets[0] is not None
        paddle.disable_static()

    def test_pir_nadam(self):
        with paddle.pir_utils.IrGuard():
            place = base.CPUPlace()
            shape = [2, 3, 8, 8]
            exe = base.Executor(place)
            train_prog = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(train_prog, startup):
                with base.unique_name.guard():
                    data = paddle.static.data(name="data", shape=shape)
                    conv_layer = paddle.nn.Conv2D(3, 8, 3)
                    conv = conv_layer(data)
                    loss = paddle.mean(conv)

                    beta1 = 0.85
                    beta2 = 0.95
                    opt = paddle.optimizer.NAdam(
                        learning_rate=1e-5,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=0.01,
                        epsilon=1e-8,
                    )
                    opt.minimize(loss)

            exe.run(startup)
            data_np = np.random.random(shape).astype('float32')
            rets = exe.run(
                train_prog, feed={"data": data_np}, fetch_list=[loss]
            )
            assert rets[0] is not None

    def test_nadam_invalid_input(self):
        paddle.disable_static()
        linear = paddle.nn.Linear(10, 10)
        with self.assertRaises(ValueError):
            _ = paddle.optimizer.NAdam(
                learning_rate=-1, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            _ = paddle.optimizer.NAdam(
                0.1, beta1=-1, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            _ = paddle.optimizer.NAdam(
                0.1, beta2=-1, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            _ = paddle.optimizer.NAdam(
                0.1, beta1=2.0, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            _ = paddle.optimizer.NAdam(
                0.1, beta2=2.0, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            _ = paddle.optimizer.NAdam(
                0.1, epsilon=-1, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            _ = paddle.optimizer.NAdam(
                0.1, momentum_decay=-1, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            _ = paddle.optimizer.NAdam(
                0.1, momentum_decay_base=-1, parameters=linear.parameters()
            )


class TestNAdamAPIGroup(TestNAdamAPI):
    def test_nadam_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        nadam = paddle.optimizer.NAdam(
            learning_rate=0.01,
            parameters=[
                {'params': linear_1.parameters()},
                {'params': linear_2.parameters(), 'weight_decay': 0.001},
            ],
            weight_decay=0.01,
        )

        for _ in range(2):
            out = linear_1(a)
            out = linear_2(out)
            out.backward()
            nadam.step()
            nadam.clear_gradients()


class TestNAdamMultiPrecision(unittest.TestCase):
    def _test_nadam_dygraph_place_amp(self, place, use_amp=False):
        paddle.disable_static()
        paddle.seed(10)
        paddle.set_device(place)

        input = paddle.randn((5, 5))

        model = paddle.nn.Linear(5, 5)

        optimizer = paddle.optimizer.NAdam(
            parameters=[
                {
                    'params': model.parameters(),
                    'weight_decay': 0.001,
                    'beta1': 0.1,
                    'beta2': 0.99,
                }
            ],
        )

        optimizer._multi_precision = use_amp

        for _ in range(2):
            if place == 'gpu' and use_amp:
                model = paddle.amp.decorate(models=model, level='O2')
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

            if place == 'gpu' and use_amp:
                with paddle.amp.auto_cast(level='O2'):
                    output = model(input)
                    loss = paddle.mean(output)
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.step(optimizer)
                optimizer.clear_grad()
            else:
                output = model(input)
                loss = paddle.mean(output)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

    def _get_places(self):
        places = ['cpu']
        if paddle.is_compiled_with_cuda():
            places.append('gpu')
        return places

    def test_main(self):
        for place in self._get_places():
            use_amp_list = [True, False]
            for use_amp in use_amp_list:
                self._test_nadam_dygraph_place_amp(place, use_amp)


class TestNAdamGroupWithLR(TestNAdamAPI):
    def test_nadam(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        nadam = paddle.optimizer.NAdam(
            learning_rate=paddle.optimizer.lr.PiecewiseDecay(
                boundaries=[3, 6], values=[0.1, 0.2, 0.3]
            ),
            parameters=[
                {
                    'params': linear_1.parameters(),
                    'learning_rate': 0.1,
                },
                {
                    'params': linear_2.parameters(),
                    'weight_decay': 0.001,
                },
            ],
            weight_decay=0.01,
        )

        for _ in range(2):
            out = linear_1(a)
            out = linear_2(out)
            out.backward()
            nadam.step()
            nadam.clear_gradients()


if __name__ == "__main__":
    unittest.main()
