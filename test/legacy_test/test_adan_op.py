# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from eager_op_test import OpTest

import paddle
from paddle import fluid
from paddle.framework import core


def adan_step(inputs, attributes):
    param = inputs['param']
    grad = inputs['grad']
    pre_grad = inputs['pregrad']
    moment1 = inputs['moment1']
    if 'moment2' in inputs:
        moment2 = inputs['moment2']
    moment3 = inputs['moment3']
    lr = inputs['learning_rate']
    beta1_pow = inputs['beta1_pow']
    beta2_pow = inputs['beta2_pow']
    beta3_pow = inputs['beta3_pow']
    epsilon = attributes['epsilon']
    if 'vanilla' in attributes:
        vanilla = attributes['vanilla']
    else:
        vanilla = False

    if 'weight_decay' in attributes:
        weight_decay = attributes["weight_decay"]

    if "no_prox" in attributes:
        no_prox = attributes["no_prox"]

    if 'beta1' in attributes:
        beta1 = attributes['beta1']
    else:
        beta1 = inputs['Beta1Tensor'][0]

    if 'beta2' in attributes:
        beta2 = attributes['beta2']
    else:
        beta2 = inputs['Beta2Tensor'][0]

    if 'beta3' in attributes:
        beta3 = attributes['beta3']
    else:
        beta3 = inputs['Beta3Tensor'][0]

    grad_diff = grad - pre_grad
    update = grad + beta2 * grad_diff
    if not vanilla:
        moment1_out = beta1 * moment1 + (1 - beta1) * grad
        moment2_out = beta2 * moment2 + (1 - beta2) * grad_diff
        moment3_out = beta3 * moment3 + (1 - beta3) * (update * update)
    else:
        moment1_out = (
            beta1 * moment1
            + (1 - beta1) * grad
            + beta2 * (1 - beta2) * grad_diff
        )
        moment2_out = None
        moment3_out = beta3 * moment3 + (1 - beta3) * (update * update)

    denom = (np.sqrt(moment3_out) / np.sqrt(1.0 - beta3_pow)) + epsilon
    if vanilla:
        update = moment1_out / (1.0 - beta1_pow) / denom
    else:
        update = (
            moment1_out / (1.0 - beta1_pow)
            + beta2 * moment2_out / (1.0 - beta2_pow)
        ) / denom

    if no_prox:
        param_out = param * (1 - lr * weight_decay) - update * lr
    else:
        param_out = (param - update * lr) / (1 + lr * weight_decay)

    return param_out, moment1_out, moment2_out, moment3_out, grad


def adan_wrapper(
    param,
    grad,
    learning_rate,
    pregrad,
    moment1,
    moment3,
    beta1_pow,
    beta2_pow,
    beta3_pow,
    moment2=None,
    master_param=None,
    beta1=0.98,
    beta2=0.92,
    beta3=0.99,
    epsilon=1e-4,
    weight_decay=0.01,
    no_prox=False,
    multi_precision=False,
    use_global_beta_pow=False,
    vanilla=False,
):
    _, _, _, _, _, _, _, _, _ = paddle._C_ops.adan_(
        param,
        grad,
        learning_rate,
        pregrad,
        moment1,
        moment3,
        beta1_pow,
        beta2_pow,
        beta3_pow,
        moment2,
        master_param,
        beta1,
        beta2,
        beta3,
        epsilon,
        weight_decay,
        no_prox,
        master_param is not None,
        use_global_beta_pow,
        vanilla,
    )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestAdan2(OpTest):
    def setUp(self):
        '''Test Adan Op with supplied attributes'''
        self.op_type = "adan"
        self.python_api = adan_wrapper
        self.python_out_sig = ['Out']
        param = np.random.uniform(-1, 1, (2, 2)).astype("float32")
        grad = np.random.uniform(-1, 1, (2, 2)).astype("float32")
        pre_grad = np.random.uniform(-1, 1, (2, 2)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (2, 2)).astype("float32")
        moment2 = np.random.random((2, 2)).astype("float32")
        moment3 = np.random.random((2, 2)).astype("float32")

        learning_rate = 0.004
        beta1 = 0.98
        beta2 = 0.92
        beta3 = 0.99
        epsilon = 1e-4
        beta1_pow = beta1**10
        beta2_pow = beta2**10
        beta3_pow = beta3**10

        self.inputs = {
            'param': param,
            'grad': grad,
            'pregrad': pre_grad,
            'moment1': moment1,
            'moment2': moment2,
            'moment3': moment3,
            'learning_rate': np.array([learning_rate]).astype("float32"),
            'beta1_pow': np.array([beta1_pow]).astype("float32"),
            'beta2_pow': np.array([beta2_pow]).astype("float32"),
            'beta3_pow': np.array([beta3_pow]).astype("float32"),
        }

        self.attrs = {
            'epsilon': epsilon,
            'beta1': beta1,
            'beta2': beta2,
            'beta3': beta3,
            "weight_decay": 0.02,
            "no_prox": False,
            "multi_precision": False,
        }

        (
            param_out,
            moment1_out,
            moment2_out,
            moment3_out,
            pre_grad_out,
        ) = adan_step(self.inputs, self.attrs)

        self.outputs = {
            'pregrad_out': pre_grad_out,
            'moment1_out': moment1_out,
            'moment2_out': moment2_out,
            'moment3_out': moment3_out,
            'param_out': param_out,
            'beta1_pow_out': np.array([beta1_pow]).astype("float32") * beta1,
            'beta2_pow_out': np.array([beta2_pow]).astype("float32") * beta2,
            'beta3_pow_out': np.array([beta3_pow]).astype("float32") * beta3,
        }

    def test_check_output(self):
        self.check_output_with_place(core.CUDAPlace(0))


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestAdanOp(unittest.TestCase):
    def test_adan_op_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        adan = paddle.optimizer.Adan(
            learning_rate=0.01,
            parameters=linear.parameters(),
            apply_decay_param_fun=lambda name: True,
            weight_decay=0.01,
        )

        for _ in range(2):
            out = linear(a)
            out.backward()
            adan.step()
            adan.clear_gradients()

    def test_adan_op_coverage(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        adan = paddle.optimizer.Adan(
            learning_rate=0.0,
            parameters=linear.parameters(),
            apply_decay_param_fun=lambda name: True,
            weight_decay=0.01,
        )
        assert adan.__str__() is not None

    def test_adan_op(self):
        paddle.enable_static()
        place = core.CUDAPlace(0)
        shape = [2, 3, 8, 8]
        exe = fluid.Executor(place)
        train_prog = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(train_prog, startup):
            with fluid.unique_name.guard():
                data = paddle.static.data(name="data", shape=shape)
                conv = paddle.static.nn.conv2d(data, 8, 3)
                loss = paddle.mean(conv)

                beta1 = paddle.static.create_global_var(
                    shape=[1], value=0.98, dtype='float32', persistable=True
                )
                beta2 = paddle.static.create_global_var(
                    shape=[1], value=0.92, dtype='float32', persistable=True
                )
                beta3 = paddle.static.create_global_var(
                    shape=[1], value=0.99, dtype='float32', persistable=True
                )
                betas = [beta1, beta2, beta3]
                opt = paddle.optimizer.Adan(
                    learning_rate=1e-5,
                    beta1=beta1,
                    beta2=beta2,
                    beta3=beta3,
                    weight_decay=0.01,
                    epsilon=1e-8,
                )
                opt.minimize(loss)

        exe.run(startup)
        data_np = np.random.random(shape).astype('float32')
        rets = exe.run(train_prog, feed={"data": data_np}, fetch_list=[loss])
        assert rets[0] is not None
        paddle.disable_static()

    def test_adan_op_invalid_input(self):
        paddle.disable_static()
        linear = paddle.nn.Linear(10, 10)
        with self.assertRaises(ValueError):
            adan = paddle.optimizer.Adan(
                0.1, beta1=-1, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            adan = paddle.optimizer.Adan(
                0.1, beta2=-1, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            adan = paddle.optimizer.Adan(
                0.1, epsilon=-1, parameters=linear.parameters()
            )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestadanOpGroup(TestAdanOp):
    def test_adan_op_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        adan = paddle.optimizer.Adan(
            learning_rate=0.01,
            parameters=[
                {'params': linear_1.parameters()},
                {'params': linear_2.parameters(), 'weight_decay': 0.001},
            ],
            apply_decay_param_fun=lambda name: True,
            weight_decay=0.01,
        )

        for _ in range(2):
            out = linear_1(a)
            out = linear_2(out)
            out.backward()
            adan.step()
            adan.clear_gradients()


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestAdanOpMultiPrecisonWithMainGrad(unittest.TestCase):
    def _test_adan_op_dygraph_place_amp_with_maingrad(
        self, place, shape, use_main_grad
    ):
        paddle.disable_static()
        paddle.seed(10)
        paddle.set_device(place)
        _weight_decay = 0.02
        no_prox = False
        find_master = True

        _epsilon = 1e-8

        _beta1 = 0.98
        _beta2 = 0.92
        _beta3 = 0.99

        lr_rate = 1e-8

        param = paddle.randn(shape).astype(paddle.bfloat16)
        master_weight = param.astype(paddle.float32)
        grad = paddle.randn(shape).astype(paddle.bfloat16)
        pregrad = paddle.randn(shape).astype(paddle.bfloat16)
        main_grad = grad.astype(paddle.float32)
        main_pregrad = pregrad.astype(paddle.float32)

        moment1 = paddle.randn(shape).astype(paddle.float32)
        moment2 = paddle.randn(shape).astype(paddle.float32)
        moment3 = paddle.randn(shape).astype(paddle.float32).abs()
        lr = paddle.zeros([1]).astype(paddle.float32)
        lr[0] = lr_rate
        beta1_pow_acc = paddle.ones([1]).astype(paddle.float32)
        beta1_pow_acc[0] = _beta1**10
        beta2_pow_acc = paddle.ones([1]).astype(paddle.float32)
        beta2_pow_acc[0] = _beta2**10
        beta3_pow_acc = paddle.ones([1]).astype(paddle.float32)
        beta3_pow_acc[0] = _beta3**10

        ref_main_pregrad = paddle.clone(main_pregrad)
        ref_param = param.astype(paddle.float32)
        ref_beta1_pow_acc = beta1_pow_acc.astype(paddle.float32)
        ref_beta2_pow_acc = beta2_pow_acc.astype(paddle.float32)
        ref_beta3_pow_acc = beta3_pow_acc.astype(paddle.float32)
        ref_moment_1 = moment1.astype(paddle.float32)
        ref_moment_2 = moment2.astype(paddle.float32)
        ref_moment_3 = moment3.astype(paddle.float32)

        # reference code
        _, _, _, _, _, _, _, _, _ = paddle._C_ops.adan_(
            ref_param,
            main_grad,
            lr,
            ref_main_pregrad,
            ref_moment_1,
            ref_moment_3,
            ref_beta1_pow_acc,
            ref_beta2_pow_acc,
            ref_beta3_pow_acc,
            ref_moment_2,
            master_weight,
            _beta1,
            _beta2,
            _beta3,
            _epsilon,
            _weight_decay,
            no_prox,
            False,
            False,
            False,
        )

        if use_main_grad:
            _, _, _, _, _, _, _, _, _ = paddle._C_ops.adan_(
                param,
                main_grad,
                lr,
                main_pregrad,
                moment1,
                moment3,
                beta1_pow_acc,
                beta2_pow_acc,
                beta3_pow_acc,
                moment2,
                master_weight,
                _beta1,
                _beta2,
                _beta3,
                _epsilon,
                _weight_decay,
                no_prox,
                find_master,
                False,
                False,
            )
            np.testing.assert_allclose(
                master_weight.numpy(),
                ref_param.numpy(),
                rtol=1e-6,
                verbose=True,
            )
            np.testing.assert_allclose(
                param.astype("float32").numpy(),
                ref_param.numpy(),
                rtol=1e-2,
                verbose=True,
            )

        else:
            _, _, _, _, _, _, _, _, _ = paddle._C_ops.adan_(
                param,
                grad,
                lr,
                pregrad,
                moment1,
                moment3,
                beta1_pow_acc,
                beta2_pow_acc,
                beta3_pow_acc,
                moment2,
                master_weight,
                _beta1,
                _beta2,
                _beta3,
                _epsilon,
                _weight_decay,
                no_prox,
                find_master,
                False,
                False,
            )
            np.testing.assert_allclose(
                master_weight.numpy(),
                ref_param.numpy(),
                rtol=1e-6,
                verbose=True,
            )
            np.testing.assert_allclose(
                param.astype("float32").numpy(),
                ref_param.numpy(),
                rtol=1e-2,
                verbose=True,
            )

    def _get_places(self):
        places = []
        if paddle.is_compiled_with_cuda():
            places.append('gpu')
        return places

    def test_main(self):
        for _ in range(10):
            shape = paddle.randint(1, 1024, [2])
            for place in self._get_places():
                use_main_grad_list = [True, False]
                for use_main_grad in use_main_grad_list:
                    self._test_adan_op_dygraph_place_amp_with_maingrad(
                        place, shape, use_main_grad
                    )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestAdanOpMultiPrecison(unittest.TestCase):
    def _test_adan_op_dygraph_place_amp(self, place, use_amp=False):
        paddle.disable_static()
        paddle.seed(10)
        paddle.set_device(place)

        input = paddle.randn((5, 5))

        model = paddle.nn.Linear(5, 5)

        optimizer = paddle.optimizer.Adan(
            parameters=[
                {
                    'params': model.parameters(),
                    'weight_decay': 0.001,
                    'beta1': 0.98,
                    'beta2': 0.92,
                    'beta3': 0.99,
                }
            ],
            multi_precision=use_amp,
        )

        for idx in range(2):
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
            # cpu 暂未开发
            if place == 'cpu':
                continue
            use_amp_list = [True, False]
            for use_amp in use_amp_list:
                self._test_adan_op_dygraph_place_amp(place, use_amp)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestAdanOpError(unittest.TestCase):
    def test_api_errors(self):
        def test_weight_decay_dtype():
            linear = paddle.nn.Linear(13, 5)
            adan = paddle.optimizer.Adan(
                learning_rate=0.01,
                parameters=linear.parameters(),
                weight_decay=1,
            )

        def test_parameters_dtype1():
            adan = paddle.optimizer.Adan(
                learning_rate=0.01,
                parameters=paddle.randn((5, 5)),
                weight_decay=0.1,
            )

        def test_parameters_dtype2():
            linear = paddle.nn.Linear(13, 5)
            adan = paddle.optimizer.Adan(
                learning_rate=0.01,
                parameters={'params': linear.parameters()},
                weight_decay=0.1,
            )

        def test_parameters_dtype3():
            adan = paddle.optimizer.Adan(
                learning_rate=0.01, parameters=None, weight_decay=0.1
            )

        def test_parameters_dtype4():
            linear = paddle.nn.Linear(13, 5)
            adan = paddle.optimizer.Adan(
                learning_rate=0.01,
                parameters={'params': set(linear.parameters())},
                weight_decay=0.1,
            )

        def test_learning_rate_dtype():
            linear = paddle.nn.Linear(13, 5)
            adan = paddle.optimizer.Adan(
                learning_rate=1,
                parameters=linear.parameters(),
                weight_decay=0.1,
            )

        def test_grad_clip_dtype():
            linear = paddle.nn.Linear(13, 5)
            adan = paddle.optimizer.Adan(
                learning_rate=0.01,
                parameters=linear.parameters(),
                weight_decay=0.1,
                grad_clip=0.1,
            )

        self.assertRaises(TypeError, test_weight_decay_dtype)
        self.assertRaises(TypeError, test_parameters_dtype1)
        self.assertRaises(TypeError, test_parameters_dtype2)
        self.assertRaises(AttributeError, test_parameters_dtype3)
        self.assertRaises(TypeError, test_parameters_dtype4)
        self.assertRaises(TypeError, test_learning_rate_dtype)
        self.assertRaises(TypeError, test_grad_clip_dtype)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestAdanOpGroupWithLR(TestAdanOp):
    def test_adan_op_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        adan = paddle.optimizer.Adan(
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
            apply_decay_param_fun=lambda name: True,
            weight_decay=0.01,
        )

        for _ in range(2):
            out = linear_1(a)
            out = linear_2(out)
            out.backward()
            adan.step()
            adan.clear_gradients()


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestAdanBetaPow2CPU(unittest.TestCase):
    def _test_adan_op_dygraph_place_amp_with_maingrad(
        self, place, shape, use_main_grad
    ):
        paddle.disable_static()
        paddle.seed(10)
        paddle.set_device(place)
        _weight_decay = 0.02
        no_prox = False
        find_master = True

        _epsilon = 1e-8

        _beta1 = 0.98
        _beta2 = 0.92
        _beta3 = 0.99

        lr_rate = 1e-8

        param = paddle.randn(shape).astype(paddle.bfloat16)
        master_weight = param.astype(paddle.float32)
        grad = paddle.randn(shape).astype(paddle.bfloat16)
        pregrad = paddle.randn(shape).astype(paddle.bfloat16)
        main_grad = grad.astype(paddle.float32)
        main_pregrad = pregrad.astype(paddle.float32)

        moment1 = paddle.randn(shape).astype(paddle.float32)
        moment2 = paddle.randn(shape).astype(paddle.float32)
        moment3 = paddle.randn(shape).astype(paddle.float32)
        lr = paddle.zeros([1]).astype(paddle.float32)
        lr[0] = lr_rate
        beta1_pow_acc = paddle.to_tensor([_beta1**10]).astype(paddle.float32)
        beta2_pow_acc = paddle.to_tensor([_beta2**10]).astype(paddle.float32)
        beta3_pow_acc = paddle.to_tensor([_beta3**10]).astype(paddle.float32)

        beta1_pow_acc = beta1_pow_acc.cpu()
        beta2_pow_acc = beta2_pow_acc.cpu()
        beta3_pow_acc = beta3_pow_acc.cpu()

        ref_main_pregrad = paddle.clone(main_pregrad)
        ref_param = param.astype(paddle.float32)
        ref_beta1_pow_acc = beta1_pow_acc.astype(paddle.float32)
        ref_beta2_pow_acc = beta2_pow_acc.astype(paddle.float32)
        ref_beta3_pow_acc = beta3_pow_acc.astype(paddle.float32)
        ref_beta1_pow_acc = ref_beta1_pow_acc.cpu()
        ref_beta2_pow_acc = ref_beta2_pow_acc.cpu()
        ref_beta3_pow_acc = ref_beta3_pow_acc.cpu()
        ref_moment_1 = moment1.astype(paddle.float32)
        ref_moment_2 = moment2.astype(paddle.float32)
        ref_moment_3 = moment3.astype(paddle.float32)

        # reference code
        _, _, _, _, _, _, _, _, _ = paddle._C_ops.adan_(
            ref_param,
            main_grad,
            lr,
            ref_main_pregrad,
            ref_moment_1,
            ref_moment_3,
            ref_beta1_pow_acc,
            ref_beta2_pow_acc,
            ref_beta3_pow_acc,
            ref_moment_2,
            master_weight,
            _beta1,
            _beta2,
            _beta3,
            _epsilon,
            _weight_decay,
            no_prox,
            False,
            False,
            False,
        )

        if use_main_grad:
            _, _, _, _, _, _, _, _, _ = paddle._C_ops.adan_(
                param,
                main_grad,
                lr,
                main_pregrad,
                moment1,
                moment3,
                beta1_pow_acc,
                beta2_pow_acc,
                beta3_pow_acc,
                moment2,
                master_weight,
                _beta1,
                _beta2,
                _beta3,
                _epsilon,
                _weight_decay,
                no_prox,
                find_master,
                False,
                False,
            )
            np.testing.assert_allclose(
                master_weight.numpy(),
                ref_param.numpy(),
                rtol=1e-6,
                verbose=True,
            )
            np.testing.assert_allclose(
                param.astype("float32").numpy(),
                ref_param.numpy(),
                rtol=1e-2,
                verbose=True,
            )

        else:
            _, _, _, _, _, _, _, _, _ = paddle._C_ops.adan_(
                param,
                grad,
                lr,
                pregrad,
                moment1,
                moment3,
                beta1_pow_acc,
                beta2_pow_acc,
                beta3_pow_acc,
                moment2,
                master_weight,
                _beta1,
                _beta2,
                _beta3,
                _epsilon,
                _weight_decay,
                no_prox,
                find_master,
                False,
                False,
            )
            np.testing.assert_allclose(
                param.astype("float32").numpy(),
                ref_param.numpy(),
                rtol=1e-2,
                verbose=True,
            )
            np.testing.assert_allclose(
                master_weight.numpy(),
                ref_param.numpy(),
                rtol=1e-6,
                verbose=True,
            )

    def _get_places(self):
        places = []
        if paddle.is_compiled_with_cuda():
            places.append('gpu')
        return places

    def test_main(self):
        for _ in range(10):
            shape = paddle.randint(1, 1024, [2])
            for place in self._get_places():
                use_main_grad_list = [True, False]
                for use_main_grad in use_main_grad_list:
                    self._test_adan_op_dygraph_place_amp_with_maingrad(
                        place, shape, use_main_grad
                    )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestAdanVanilla(OpTest):
    def setUp(self):
        '''Test Adan Op with supplied attributes'''
        self.op_type = "adan"
        self.python_api = adan_wrapper
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.
        param = np.random.uniform(-1, 1, (2, 2)).astype("float32")
        grad = np.random.uniform(-1, 1, (2, 2)).astype("float32")
        pre_grad = np.random.uniform(-1, 1, (2, 2)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (2, 2)).astype("float32")
        moment3 = np.random.random((2, 2)).astype("float32")

        learning_rate = 0.004
        beta1 = 0.98
        beta2 = 0.92
        beta3 = 0.99
        epsilon = 1e-4
        beta1_pow = beta1**10
        beta2_pow = beta2**10
        beta3_pow = beta3**10

        self.inputs = {
            'param': param,
            'grad': grad,
            'pregrad': pre_grad,
            'moment1': moment1,
            'moment3': moment3,
            'learning_rate': np.array([learning_rate]).astype("float32"),
            'beta1_pow': np.array([beta1_pow]).astype("float32"),
            'beta2_pow': np.array([beta2_pow]).astype("float32"),
            'beta3_pow': np.array([beta3_pow]).astype("float32"),
        }

        self.attrs = {
            'epsilon': epsilon,
            'beta1': beta1,
            'beta2': beta2,
            'beta3': beta3,
            "weight_decay": 0.02,
            "no_prox": False,
            "multi_precision": False,
            "vanilla": True,
        }

        (
            param_out,
            moment1_out,
            moment2_out,
            moment3_out,
            pre_grad_out,
        ) = adan_step(self.inputs, self.attrs)

        self.outputs = {
            'pregrad_out': pre_grad_out,
            'moment1_out': moment1_out,
            'moment3_out': moment3_out,
            'param_out': param_out,
            'beta1_pow_out': np.array([beta1_pow]).astype("float32") * beta1,
            'beta2_pow_out': np.array([beta2_pow]).astype("float32") * beta2,
            'beta3_pow_out': np.array([beta3_pow]).astype("float32") * beta3,
        }

    def test_check_output(self):
        self.check_output_with_place(core.CUDAPlace(0))


if __name__ == "__main__":
    unittest.main()
