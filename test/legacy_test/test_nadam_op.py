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

import os
import unittest
from copy import deepcopy

import numpy as np
from op_test import OpTest

import paddle
from paddle import base
from paddle.framework import core

RTOL = 1e-06
ATOL = 1e-06


def nadam_step(inputs, attributes, dtype='float32'):
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

    momentum_decay_pow *= 0.96
    beta2_pow *= beta2

    mu_t = beta1 * (1.0 - 0.5 * (momentum_decay_pow**momentum_decay))
    mu_t_1 = beta1 * (
        1.0
        - 0.5 * (momentum_decay_pow**momentum_decay) * (0.96**momentum_decay)
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
    return (
        param.astype(dtype),
        momentum_decay_pow.astype(dtype),
        beta2_pow.astype(dtype),
        mu_product.astype(dtype),
        moment1.astype(dtype),
        moment2.astype(dtype),
    )


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
        multi_precision,
    )


class TestNAdamOp(OpTest):
    def _init_param(self):
        self.beta1 = 0.78
        self.beta2 = 0.915
        self.epsilon = 1e-8
        self.momentum_decay = 0.004

    def setUp(self):
        '''Test NAdam Op with supplied attributes'''
        np.random.seed(2024)

        self.op_type = "nadam"
        self.python_api = nadam_wrapper
        self.python_out_sig = ['out']

        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        learning_rate = np.array(0.003).astype("float32")

        self._init_param()

        # accumulators
        momentum_decay_pow = (np.ones((102, 105)) * (0.96**3)).astype("float32")
        # use beta1 to fake mu_product
        mu_product = (np.ones((102, 105)) * (self.beta1**3)).astype("float32")
        beta2_pow = (np.ones((102, 105)) * (self.beta2**3)).astype("float32")
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
            "epsilon": self.epsilon,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "momentum_decay": self.momentum_decay,
        }

        (
            param_out,
            momentum_decay_pow_out,
            beta2_pow_out,
            mu_product_out,
            moment1_out,
            moment2_out,
        ) = nadam_step(deepcopy(self.inputs), deepcopy(self.attrs))

        self.outputs = {
            "param_out": param_out,
            "momentum_decay_pow_out": momentum_decay_pow_out,
            "beta2_pow_out": beta2_pow_out,
            "mu_product_out": mu_product_out,
            "moment1_out": moment1_out,
            "moment2_out": moment2_out,
        }

    def test_check_output(self):
        self.check_output(check_pir=True, rtol=RTOL, atol=ATOL)


class TestNAdamOpWithDefault(TestNAdamOp):
    def _init_param(self):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1.0e-8
        self.momentum_decay = 0.004


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestNAdamOpGPU(TestNAdamOp):
    def test_check_output(self):
        self.check_output_with_place(
            core.CUDAPlace(0), check_pir=True, rtol=RTOL, atol=ATOL
        )


class TestNAdamOpMultipleSteps(TestNAdamOp):
    num_steps = 10

    def test_check_output(self):
        for _ in range(self.num_steps):
            (
                param_out,
                momentum_decay_pow_out,
                beta2_pow_out,
                mu_product_out,
                moment1_out,
                moment2_out,
            ) = nadam_step(deepcopy(self.inputs), deepcopy(self.attrs))

            self.outputs = {
                "param_out": param_out,
                "momentum_decay_pow_out": momentum_decay_pow_out,
                "beta2_pow_out": beta2_pow_out,
                "mu_product_out": mu_product_out,
                "moment1_out": moment1_out,
                "moment2_out": moment2_out,
            }

            # Verify output for this step
            self.check_output()

            # Output of this step becomes input for next step
            self.inputs['param'] = param_out
            self.inputs['momentum_decay_pow'] = momentum_decay_pow_out
            self.inputs['beta2_pow'] = beta2_pow_out
            self.inputs['mu_product'] = mu_product_out
            self.inputs['moment1'] = moment1_out
            self.inputs['moment2'] = moment2_out

            # Randomize gradient for next step
            self.inputs['grad'] = np.random.uniform(-1, 1, (102, 105)).astype(
                "float32"
            )


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
                hidden = paddle.static.nn.fc(x=data, size=10)
                loss = paddle.mean(hidden)

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
                    hidden = paddle.static.nn.fc(x=data, size=10)
                    loss = paddle.mean(hidden)

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


class TestNAdamAPIWeightDecay(unittest.TestCase):
    def test_weight_decay_int(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        nadam = paddle.optimizer.NAdam(
            learning_rate=0.01,
            parameters=linear.parameters(),
            weight_decay=1,
        )

        for _ in range(2):
            out = linear(a)
            out.backward()
            nadam.step()
            nadam.clear_gradients()


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
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            places.append('cpu')
        if paddle.is_compiled_with_cuda():
            places.append('gpu')
        return places

    def test_main(self):
        for place in self._get_places():
            use_amp_list = [True, False]
            for use_amp in use_amp_list:
                self._test_nadam_dygraph_place_amp(place, use_amp)


class TestNdamaxMultiPrecision2_0(unittest.TestCase):
    def dygraph_nadam_mp(self, mp, use_amp):
        paddle.disable_static()
        paddle.seed(100)
        paddle.set_device('gpu')
        input = paddle.randn((2, 2))
        model = paddle.nn.Linear(2, 2)
        optimizer = paddle.optimizer.NAdam(0.1, parameters=model.parameters())
        optimizer._multi_precision = mp
        if use_amp:
            model = paddle.amp.decorate(models=model, level='O2')
            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        for idx in range(5):
            if use_amp:
                with paddle.amp.auto_cast(level='O2'):
                    output = model(input)
                    loss = paddle.mean(output)
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.minimize(optimizer, scaled)
                optimizer.clear_grad()
            else:
                output = model(input)
                loss = paddle.mean(output)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

        return output, model.parameters()

    def static_nadam_mp(self, mp, use_amp):
        paddle.enable_static()
        paddle.seed(2024)
        with paddle.pir_utils.OldIrGuard():
            exe = paddle.static.Executor('gpu')
            train_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            optimizer = paddle.optimizer.NAdam(0.1)
            optimizer._multi_precision = mp
            if use_amp:
                optimizer = paddle.static.amp.decorate(
                    optimizer,
                    init_loss_scaling=128.0,
                    use_dynamic_loss_scaling=True,
                    use_pure_fp16=True,
                    use_fp16_guard=False,
                )
            with paddle.static.program_guard(train_program, startup_program):
                if use_amp:
                    data = paddle.static.data(
                        shape=[2, 2], name='X', dtype='float16'
                    )
                else:
                    data = paddle.static.data(
                        shape=[2, 2], name='X', dtype='float32'
                    )
                hidden = paddle.static.nn.fc(x=data, size=10)
                loss = paddle.mean(hidden)
                optimizer.minimize(loss)
            exe.run(startup_program)

            np.random.seed(2024)
            if use_amp:
                optimizer.amp_init(
                    place=paddle.CUDAPlace(0),
                    scope=paddle.static.global_scope(),
                )
                x = np.random.random(size=(2, 2)).astype('float16')
            else:
                x = np.random.random(size=(2, 2)).astype('float32')
            out = []
            for idx in range(5):
                (loss_data,) = exe.run(
                    train_program, feed={"X": x}, fetch_list=[loss.name]
                )
                out.append(loss_data)

            return out

    def pir_nadam_mp(self, mp, use_amp):
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            paddle.seed(2024)
            exe = paddle.static.Executor('gpu')
            train_program = paddle.static.Program()
            startup_program = paddle.static.Program()

            with paddle.static.program_guard(train_program, startup_program):
                model = paddle.nn.Linear(2, 10)
                optimizer = paddle.optimizer.NAdam(
                    0.1, parameters=model.parameters()
                )
                if use_amp:
                    data = paddle.static.data(
                        shape=[2, 2], name='X', dtype='float16'
                    )
                    model, optimizer = paddle.amp.decorate(
                        models=model,
                        optimizers=optimizer,
                        level='O2',
                        master_weight=mp,
                    )
                    scaler = paddle.amp.GradScaler(init_loss_scaling=128.0)
                    with paddle.amp.auto_cast(
                        level='O2', dtype="float16", use_promote=True
                    ):
                        output = model(data)
                        loss = paddle.mean(output)
                    scaled = scaler.scale(loss)
                    scaler.minimize(optimizer, scaled)
                else:
                    data = paddle.static.data(
                        shape=[2, 2], name='X', dtype='float32'
                    )
                    output = model(data)
                    loss = paddle.mean(output)
                    optimizer.minimize(loss)
            exe.run(startup_program)

            np.random.seed(2024)
            if use_amp:
                x = np.random.random(size=(2, 2)).astype('float16')
            else:
                x = np.random.random(size=(2, 2)).astype('float32')
            out = []
            for idx in range(5):
                (loss_data,) = exe.run(
                    train_program, feed={"X": x}, fetch_list=[loss]
                )
                out.append(loss_data)

            return out

    def static_nadam_amp_o2_without_scaler(self):
        paddle.enable_static()
        paddle.seed(2024)
        with paddle.pir_utils.IrGuard():
            train_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(train_program, startup_program):
                exe = paddle.static.Executor('gpu')
                linear = paddle.nn.Linear(2, 10)
                optimizer = paddle.optimizer.NAdam(
                    0.1, parameters=linear.parameters()
                )
                linear, optimizer = paddle.amp.decorate(
                    optimizers=optimizer,
                    models=linear,
                    level='O2',
                )
                data = paddle.static.data(
                    shape=[2, 2], name='X', dtype='float32'
                )
                with paddle.amp.auto_cast(
                    level='O2', dtype='float16', use_promote=True
                ):
                    out = linear(data)
                    loss = paddle.mean(out)
                optimizer.minimize(loss)
                exe.run(startup_program)

                np.random.seed(2024)

                x = np.random.random(size=(2, 2)).astype('float32')
                out = []
                for idx in range(5):
                    (loss_data,) = exe.run(
                        train_program, feed={"X": x}, fetch_list=[loss]
                    )
                    out.append(loss_data)

                return out

    def test_main(self):
        if not paddle.is_compiled_with_cuda():
            return
        "Test dygraph mode"
        output1_dy, params1_dy = self.dygraph_nadam_mp(use_amp=True, mp=True)
        output2_dy, params2_dy = self.dygraph_nadam_mp(use_amp=False, mp=False)
        np.testing.assert_allclose(
            output1_dy.astype('float32').numpy(),
            output2_dy.astype('float32').numpy(),
            rtol=1e-05,
            atol=0.1,
        )
        for idx in range(len(params1_dy)):
            np.testing.assert_allclose(
                params1_dy[idx].astype('float32').numpy(),
                params2_dy[idx].astype('float32').numpy(),
                rtol=1e-05,
                atol=0.1,
            )
        "Test static mode"
        output1_st = self.static_nadam_mp(use_amp=True, mp=True)
        output2_st = self.static_nadam_mp(use_amp=False, mp=False)
        output3_st = self.static_nadam_amp_o2_without_scaler()
        for idx in range(len(output1_st)):
            np.testing.assert_allclose(
                output1_st[idx].astype('float32'),
                output2_st[idx].astype('float32'),
                rtol=1e-05,
                atol=0.1,
            )
        "Test pir mode"
        output1_pir = self.pir_nadam_mp(use_amp=True, mp=True)
        output2_pir = self.pir_nadam_mp(use_amp=False, mp=False)
        for idx in range(len(output1_st)):
            np.testing.assert_allclose(
                output1_pir[idx].astype('float32'),
                output2_pir[idx].astype('float32'),
                rtol=1e-05,
                atol=0.1,
            )

        for idx in range(len(output1_st)):
            np.testing.assert_allclose(
                output1_st[idx].astype('float32'),
                output3_st[idx].astype('float32'),
                rtol=1e-05,
                atol=0.1,
            )


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


def get_places():
    places = []
    if (
        os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
        in ['1', 'true', 'on']
        or not base.is_compiled_with_cuda()
    ):
        places.append(base.CPUPlace())
    if base.is_compiled_with_cuda():
        places.append(base.CUDAPlace(0))
    return places


def main_test_func(place, dtype):
    paddle.enable_static()
    main = base.Program()
    startup = base.Program()
    with base.program_guard(main, startup):
        with base.scope_guard(base.Scope()):
            x = paddle.static.data(name='x', shape=[None, 13], dtype=dtype)
            y = paddle.static.data(name='y', shape=[None, 1], dtype=dtype)
            y_predict = paddle.static.nn.fc(x, size=1)
            cost = paddle.nn.functional.square_error_cost(
                input=y_predict, label=y
            )
            avg_cost = paddle.mean(cost)

            nadam_optimizer = paddle.optimizer.NAdam(0.01)
            nadam_optimizer.minimize(avg_cost)

            fetch_list = [avg_cost]
            train_reader = list(
                zip(
                    np.random.rand(101, 13),
                    np.random.randint(12, size=(101, 1)),
                )
            )
            feeder = base.DataFeeder(place=place, feed_list=[x, y])
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            for data in train_reader:
                exe.run(main, feed=feeder.feed([data]), fetch_list=fetch_list)

    paddle.disable_static()


class NAdamFp32Test(unittest.TestCase):
    def setUp(self):
        self.dtype = 'float32'

    def test_main(self):
        for p in get_places():
            main_test_func(p, self.dtype)


class NAdamFp64Test(NAdamFp32Test):
    def setUp(self):
        self.dtype = 'float64'


if __name__ == "__main__":
    unittest.main()
