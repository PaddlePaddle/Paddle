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

import random
import unittest
from functools import partial

import numpy as np
from op_test import OpTest

import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import _test_eager_guard
from paddle.framework import core


def adamw_step(inputs, attributes):
    param = inputs['Param']
    grad = inputs['Grad']
    moment1 = inputs['Moment1']
    moment2 = inputs['Moment2']
    lr = inputs['LearningRate']
    beta1_pow = inputs['Beta1Pow']
    beta2_pow = inputs['Beta2Pow']

    epsilon = attributes['epsilon']

    if 'lr_ratio' in attributes:
        lr = lr * attributes['lr_ratio']

    if attributes["with_decay"]:
        coeff = attributes["coeff"]
        decay = 1.0 - lr * coeff
        param2 = param * decay
        param = param2.copy()

    if 'beta1' in attributes:
        beta1 = attributes['beta1']
    else:
        beta1 = inputs['Beta1Tensor'][0]
    if 'beta2' in attributes:
        beta2 = attributes['beta2']
    else:
        beta2 = inputs['Beta2Tensor'][0]

    moment1_out = beta1 * moment1 + (1 - beta1) * grad
    moment2_out = beta2 * moment2 + (1 - beta2) * np.square(grad)
    denom = (np.sqrt(moment2_out) / np.sqrt(1.0 - beta2_pow)) + epsilon
    param_out = param + ((moment1_out / denom) * (-(lr / (1.0 - beta1_pow))))
    return param_out, moment1_out, moment2_out


class TestAdamW(OpTest):
    def setUp(self):
        '''Test AdamW Op with supplied attributes'''
        self.op_type = "adamw"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.004
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32"),
        }

        self.attrs = {
            'epsilon': epsilon,
            'beta1': beta1,
            'beta2': beta2,
            "coeff": 0.5,
            "with_decay": True,
        }

        param_out, moment1_out, moment2_out = adamw_step(
            self.inputs, self.attrs
        )

        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'ParamOut': param_out,
            'Beta1PowOut': np.array([beta1_pow]).astype("float32") * beta1,
            'Beta2PowOut': np.array([beta2_pow]).astype("float32") * beta2,
        }

    def test_check_output(self):
        self.check_output()


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestAdamW2(OpTest):
    def setUp(self):
        '''Test AdamW Op with supplied attributes'''
        self.op_type = "adamw"
        param = np.random.uniform(-1, 1, (2, 2)).astype("float32")
        grad = np.random.uniform(-1, 1, (2, 2)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (2, 2)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((2, 2)).astype("float32")

        learning_rate = 0.004
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32"),
        }

        self.attrs = {
            'epsilon': epsilon,
            'beta1': beta1,
            'beta2': beta2,
            "lr_ratio": 0.1,
            "coeff": 0.5,
            "with_decay": True,
        }

        param_out, moment1_out, moment2_out = adamw_step(
            self.inputs, self.attrs
        )

        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'ParamOut': param_out,
            'Beta1PowOut': np.array([beta1_pow]).astype("float32") * beta1,
            'Beta2PowOut': np.array([beta2_pow]).astype("float32") * beta2,
        }

    def test_check_output(self):
        self.check_output_with_place(core.CUDAPlace(0))


class TestAdamWOp(unittest.TestCase):
    def test_adamw_op_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        adam = paddle.optimizer.AdamW(
            learning_rate=0.01,
            parameters=linear.parameters(),
            apply_decay_param_fun=lambda name: True,
            weight_decay=0.01,
        )

        for _ in range(2):
            out = linear(a)
            out.backward()
            adam.step()
            adam.clear_gradients()

    def test_adamw_op_coverage(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        adam = paddle.optimizer.AdamW(
            learning_rate=0.0,
            parameters=linear.parameters(),
            apply_decay_param_fun=lambda name: True,
            weight_decay=0.01,
        )
        assert adam.__str__() is not None

    def test_adamw_op(self):
        paddle.enable_static()
        place = fluid.CPUPlace()
        shape = [2, 3, 8, 8]
        exe = fluid.Executor(place)
        train_prog = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(train_prog, startup):
            with fluid.unique_name.guard():
                data = fluid.data(name="data", shape=shape)
                conv = fluid.layers.conv2d(data, 8, 3)
                loss = paddle.mean(conv)

                beta1 = paddle.static.create_global_var(
                    shape=[1], value=0.85, dtype='float32', persistable=True
                )
                beta2 = paddle.static.create_global_var(
                    shape=[1], value=0.95, dtype='float32', persistable=True
                )
                betas = [beta1, beta2]
                opt = paddle.optimizer.AdamW(
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

    def test_adamw_op_invalid_input(self):
        paddle.disable_static()
        linear = paddle.nn.Linear(10, 10)
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.AdamW(
                0.1, beta1=-1, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.AdamW(
                0.1, beta2=-1, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.AdamW(
                0.1, epsilon=-1, parameters=linear.parameters()
            )

    def test_api_eager_dygraph(self):
        with _test_eager_guard():
            self.test_adamw_op_dygraph()
            self.test_adamw_op_invalid_input()


class TestAdamWOpGroup(TestAdamWOp):
    def test_adamw_op_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        adam = paddle.optimizer.AdamW(
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
            adam.step()
            adam.clear_gradients()


class TestAdamWOpMultiPrecison(unittest.TestCase):
    def _test_adamw_op_dygraph_place_amp(self, place, use_amp=False):
        paddle.disable_static()
        paddle.seed(10)
        paddle.set_device(place)

        input = paddle.randn((5, 5))

        model = paddle.nn.Linear(5, 5)

        optimizer = paddle.optimizer.AdamW(
            parameters=[
                {
                    'params': model.parameters(),
                    'weight_decay': 0.001,
                    'beta1': 0.1,
                    'beta2': 0.99,
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
            use_amp_list = [True, False]
            for use_amp in use_amp_list:
                self._test_adamw_op_dygraph_place_amp(place, use_amp)


class TestAdamWOpError(unittest.TestCase):
    def test_api_errors(self):
        def test_weight_decay_dtype():
            linear = paddle.nn.Linear(13, 5)
            adam = paddle.optimizer.AdamW(
                learning_rate=0.01,
                parameters=linear.parameters(),
                weight_decay=1,
            )

        def test_parameters_dtype1():
            adam = paddle.optimizer.AdamW(
                learning_rate=0.01,
                parameters=paddle.randn((5, 5)),
                weight_decay=0.1,
            )

        def test_parameters_dtype2():
            linear = paddle.nn.Linear(13, 5)
            adam = paddle.optimizer.AdamW(
                learning_rate=0.01,
                parameters={'params': linear.parameters()},
                weight_decay=0.1,
            )

        def test_parameters_dtype3():
            adam = paddle.optimizer.AdamW(
                learning_rate=0.01, parameters=None, weight_decay=0.1
            )

        def test_parameters_dtype4():
            linear = paddle.nn.Linear(13, 5)
            adam = paddle.optimizer.AdamW(
                learning_rate=0.01,
                parameters={'params': set(linear.parameters())},
                weight_decay=0.1,
            )

        def test_learning_rate_dtype():
            linear = paddle.nn.Linear(13, 5)
            adam = paddle.optimizer.AdamW(
                learning_rate=1,
                parameters=linear.parameters(),
                weight_decay=0.1,
            )

        def test_grad_clip_dtype():
            linear = paddle.nn.Linear(13, 5)
            adam = paddle.optimizer.AdamW(
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


class TestAdamWOpGroupWithLR(TestAdamWOp):
    def test_adamw_op_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        adam = paddle.optimizer.AdamW(
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
            adam.step()
            adam.clear_gradients()


def simple_lr_setting(param, decay_rate, n_layers):
    if "fc_0" in param.name or "linear_1" in param.name:
        depth = int(param.name.split("_")[2]) + 1
    elif "fc_1" in param.name or "linear_2" in param.name:
        depth = int(param.name.split("_")[2]) + 2
    else:
        depth = 0

    return decay_rate ** (n_layers + 2 - depth)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestAdamWOpLayerwiseLR(TestAdamWOp):
    def setUp(self):
        random.seed(2022)
        np.random.seed(2022)
        paddle.seed(2022)

    def test_adamw_op_dygraph(self):
        paddle.disable_static()
        linear1 = paddle.nn.Linear(
            13, 8, bias_attr=paddle.nn.initializer.Constant(value=1.0)
        )
        linear2 = paddle.nn.Linear(
            8, 5, bias_attr=paddle.nn.initializer.Constant(value=1.0)
        )

        # fix the linear name, simple_lr_setting function will use the name
        linear1.weight.name = "linear_1.w_0"
        linear1.bias.name = "linear_1.b_0"
        linear2.weight.name = "linear_2.w_0"
        linear2.bias.name = "linear_2.b_0"

        fc1_w = np.array(linear1.weight)
        fc1_w_mon1 = np.zeros_like(fc1_w)
        fc1_w_mon2 = np.zeros_like(fc1_w)
        fc1_b = np.array(linear1.bias)
        fc1_b_mon1 = np.zeros_like(fc1_b)
        fc1_b_mon2 = np.zeros_like(fc1_b)

        fc2_w = np.array(linear2.weight)
        fc2_w_mon1 = np.zeros_like(fc2_w)
        fc2_w_mon2 = np.zeros_like(fc2_w)
        fc2_b = np.array(linear2.bias)
        fc2_b_mon1 = np.zeros_like(fc2_b)
        fc2_b_mon2 = np.zeros_like(fc2_b)

        simple_lr_fun = partial(simple_lr_setting, decay_rate=0.8, n_layers=2)
        learning_rate = 0.001
        weight_decay = 0.01
        beta1 = 0.9
        beta2 = 0.999

        opt = paddle.optimizer.AdamW(
            learning_rate=learning_rate,
            parameters=[
                {'params': linear1.parameters()},
                {
                    'params': linear2.parameters(),
                },
            ],
            apply_decay_param_fun=lambda name: True,
            weight_decay=weight_decay,
            lr_ratio=simple_lr_fun,
        )

        def get_numpy_output(param, grad, moment1, moment2, lr_ratio, t):
            np_inputs = {
                'Param': param,
                'Grad': grad,
                'Moment1': moment1,
                'Moment2': moment2,
                'LearningRate': np.array([learning_rate]).astype("float32"),
                'Beta1Pow': np.array([beta1**t]).astype("float32"),
                'Beta2Pow': np.array([beta2**t]).astype("float32"),
            }

            np_attrs = {
                'epsilon': 1e-8,
                'beta1': beta1,
                'beta2': beta2,
                "lr_ratio": lr_ratio,
                "coeff": weight_decay,
                "with_decay": True,
            }
            param_out, moment1_out, moment2_out = adamw_step(
                np_inputs, np_attrs
            )
            return param_out, moment1_out, moment2_out

        for i in range(5):
            a = paddle.to_tensor(
                np.random.uniform(-1, 1, (2, 13)).astype("float32")
            )
            a1 = linear1(a)
            out = linear2(a1)
            out = paddle.mean(out)
            out.backward()

            fc1_w, fc1_w_mon1, fc1_w_mon2 = get_numpy_output(
                fc1_w,
                np.array(linear1.weight.grad),
                fc1_w_mon1,
                fc1_w_mon2,
                simple_lr_fun(linear1.weight),
                i + 1,
            )
            fc1_b, fc1_b_mon1, fc1_b_mon2 = get_numpy_output(
                fc1_b,
                np.array(linear1.bias.grad),
                fc1_b_mon1,
                fc1_b_mon2,
                simple_lr_fun(linear1.bias),
                i + 1,
            )
            fc2_w, fc2_w_mon1, fc2_w_mon2 = get_numpy_output(
                fc2_w,
                np.array(linear2.weight.grad),
                fc2_w_mon1,
                fc2_w_mon2,
                simple_lr_fun(linear2.weight),
                i + 1,
            )
            fc2_b, fc2_b_mon1, fc2_b_mon2 = get_numpy_output(
                fc2_b,
                np.array(linear2.bias.grad),
                fc2_b_mon1,
                fc2_b_mon2,
                simple_lr_fun(linear2.bias),
                i + 1,
            )

            opt.step()
            opt.clear_gradients()

            np.testing.assert_allclose(linear1.weight.numpy(), fc1_w, rtol=1e-6)
            np.testing.assert_allclose(linear1.bias.numpy(), fc1_b, rtol=1e-6)
            np.testing.assert_allclose(linear2.weight.numpy(), fc2_w, rtol=1e-6)
            np.testing.assert_allclose(linear2.bias.numpy(), fc2_b, rtol=1e-6)

    def test_adamw_op(self):
        paddle.enable_static()
        place = fluid.CUDAPlace(0)

        learning_rate = 0.0001
        beta1 = 0.85
        beta2 = 0.95
        weight_decay = 0.01
        epsilon = 1e-8

        train_prog = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(train_prog, startup):
            with fluid.unique_name.guard():
                x = fluid.data(name='x', shape=[None, 10], dtype='float32')
                y = fluid.data(name='y', shape=[None, 1], dtype='float32')

                weight_attr1 = paddle.framework.ParamAttr(name="linear_0.w_0")
                bias_attr1 = paddle.framework.ParamAttr(
                    name="linear_0.b_0",
                    initializer=paddle.nn.initializer.Constant(value=1.0),
                )
                weight_attr2 = paddle.framework.ParamAttr(name="linear_1.w_0")
                bias_attr2 = paddle.framework.ParamAttr(
                    name="linear_1.b_0",
                    initializer=paddle.nn.initializer.Constant(value=1.0),
                )
                linear1 = paddle.nn.Linear(
                    10, 32, weight_attr=weight_attr1, bias_attr=bias_attr1
                )
                linear2 = paddle.nn.Linear(
                    32, 1, weight_attr=weight_attr2, bias_attr=bias_attr2
                )

                out = linear1(x)
                out = linear2(out)

                fc1_w_mon1 = np.zeros((linear1.weight.shape)).astype("float32")
                fc1_w_mon2 = np.zeros((linear1.weight.shape)).astype("float32")
                fc1_b_mon1 = np.zeros((linear1.bias.shape)).astype("float32")
                fc1_b_mon2 = np.zeros((linear1.bias.shape)).astype("float32")
                fc2_w_mon1 = np.zeros((linear2.weight.shape)).astype("float32")
                fc2_w_mon2 = np.zeros((linear2.weight.shape)).astype("float32")
                fc2_b_mon1 = np.zeros((linear2.bias.shape)).astype("float32")
                fc2_b_mon2 = np.zeros((linear2.bias.shape)).astype("float32")

                cost = paddle.nn.functional.square_error_cost(
                    input=out, label=y
                )
                avg_cost = paddle.mean(cost)

                simple_lr_fun = partial(
                    simple_lr_setting, decay_rate=0.8, n_layers=2
                )

                opt = paddle.optimizer.AdamW(
                    learning_rate=learning_rate,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    epsilon=epsilon,
                    lr_ratio=simple_lr_fun,
                )
                opt.minimize(avg_cost)

        def get_numpy_output(param, grad, moment1, moment2, lr_ratio, t):
            np_inputs = {
                'Param': param,
                'Grad': grad,
                'Moment1': moment1,
                'Moment2': moment2,
                'LearningRate': np.array([learning_rate]).astype("float32"),
                'Beta1Pow': np.array([beta1**t]).astype("float32"),
                'Beta2Pow': np.array([beta2**t]).astype("float32"),
            }

            np_attrs = {
                'epsilon': epsilon,
                'beta1': beta1,
                'beta2': beta2,
                "lr_ratio": lr_ratio,
                "coeff": weight_decay,
                "with_decay": True,
            }
            param_out, moment1_out, moment2_out = adamw_step(
                np_inputs, np_attrs
            )
            return param_out, moment1_out, moment2_out

        fetch_list1 = [
            "linear_0.w_0",
            "linear_0.b_0",
            "linear_1.w_0",
            "linear_1.b_0",
        ]
        fetch_list2 = [
            "linear_0.w_0",
            "linear_0.w_0@GRAD",
            "linear_0.b_0",
            "linear_0.b_0@GRAD",
            "linear_1.w_0",
            "linear_1.w_0@GRAD",
            "linear_1.b_0",
            "linear_1.b_0@GRAD",
        ]

        exe = fluid.Executor(place)
        exe.run(startup)
        test_prog = train_prog.clone(for_test=True)

        for i in range(5):
            inputs = np.random.random(size=[8, 10]).astype('float32')
            outputs = np.random.random(size=[8, 1]).astype('float32')

            param = exe.run(
                test_prog,
                feed={"x": inputs, "y": outputs},
                fetch_list=fetch_list1,
            )
            params_and_gras = exe.run(
                train_prog,
                feed={"x": inputs, "y": outputs},
                fetch_list=fetch_list2,
            )

            fc1_w = param[0]
            fc1_w_grad = params_and_gras[1]
            fc1_b = param[1]
            fc1_b_grad = params_and_gras[3]
            fc2_w = param[2]
            fc2_w_grad = params_and_gras[5]
            fc2_b = param[3]
            fc2_b_grad = params_and_gras[7]

            fc1_w, fc1_w_mon1, fc1_w_mon2 = get_numpy_output(
                fc1_w,
                fc1_w_grad,
                fc1_w_mon1,
                fc1_w_mon2,
                simple_lr_fun(linear1.weight),
                i + 1,
            )
            fc1_b, fc1_b_mon1, fc1_b_mon2 = get_numpy_output(
                fc1_b,
                fc1_b_grad,
                fc1_b_mon1,
                fc1_b_mon2,
                simple_lr_fun(linear1.bias),
                i + 1,
            )
            fc2_w, fc2_w_mon1, fc2_w_mon2 = get_numpy_output(
                fc2_w,
                fc2_w_grad,
                fc2_w_mon1,
                fc2_w_mon2,
                simple_lr_fun(linear2.weight),
                i + 1,
            )
            fc2_b, fc2_b_mon1, fc2_b_mon2 = get_numpy_output(
                fc2_b,
                fc2_b_grad,
                fc2_b_mon1,
                fc2_b_mon2,
                simple_lr_fun(linear2.bias),
                i + 1,
            )

            np.testing.assert_allclose(params_and_gras[0], fc1_w, rtol=1e-6)
            np.testing.assert_allclose(params_and_gras[2], fc1_b, rtol=1e-6)
            np.testing.assert_allclose(params_and_gras[4], fc2_w, rtol=1e-6)
            np.testing.assert_allclose(params_and_gras[6], fc2_b, rtol=1e-6)

        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
