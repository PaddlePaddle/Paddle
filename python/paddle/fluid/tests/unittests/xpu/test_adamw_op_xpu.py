# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import sys

sys.path.append("..")

import unittest

import numpy as np
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

import paddle
import paddle.fluid as fluid


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
    lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)
    param_out = param - lr_t * (moment1_out / (np.sqrt(moment2_out) + epsilon))
    return param_out, moment1_out, moment2_out


def simple_lr_setting(param, decay_rate, n_layers):
    if "fc_0" in param.name or "linear_1" in param.name:
        depth = int(param.name.split("_")[2]) + 1
    elif "fc_1" in param.name or "linear_2" in param.name:
        depth = int(param.name.split("_")[2]) + 2
    else:
        depth = 0

    return decay_rate ** (n_layers + 2 - depth)


class XPUTestAdamwOp1(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'adamw'
        self.use_dynamic_create_class = False

    class TestAdamW(XPUOpTest):
        def setUp(self):
            # Test AdamW Op with supplied attributes
            self.op_type = "adamw"
            self.init_shape()
            self.dtype = self.in_type_str
            param = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            grad = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            moment1 = np.random.uniform(-1, 1, self.shape).astype("float32")
            # The second moment is positive
            moment2 = np.random.random(self.shape).astype("float32")

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

        def init_shape(self):
            self.shape = [102, 105]

        def test_check_output(self):
            paddle.enable_static()
            self.check_output_with_place(place=paddle.XPUPlace(0))

    class TestAdamW2(TestAdamW):
        def init_shape(self):
            self.shape = [
                1000,
            ]

    class TestAdamW3(TestAdamW):
        def init_shape(self):
            self.shape = [200, 3000]


class XPUTestAdamwOp2(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'adamw'
        self.use_dynamic_create_class = False

    class TestAdamWOp(unittest.TestCase):
        def test_adamw_op_dygraph(self):
            paddle.disable_static()
            value = np.arange(26).reshape(2, 13).astype(self.in_type_str)
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
            value = np.arange(26).reshape(2, 13).astype(self.in_type_str)
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
            place = fluid.XPUPlace(0)
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
                        shape=[1],
                        value=0.85,
                        dtype=self.in_type_str,
                        persistable=True,
                    )
                    beta2 = paddle.static.create_global_var(
                        shape=[1],
                        value=0.95,
                        dtype=self.in_type_str,
                        persistable=True,
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
            data_np = np.random.random(shape).astype(self.in_type_str)
            rets = exe.run(
                train_prog, feed={"data": data_np}, fetch_list=[loss]
            )
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

    class TestAdamWOpGroup(TestAdamWOp):
        def test_adamw_op_dygraph(self):
            paddle.disable_static()
            value = np.arange(26).reshape(2, 13).astype(self.in_type_str)
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

    class TestAdamWOpGroupWithLR(TestAdamWOp):
        def test_adamw_op_dygraph(self):
            paddle.disable_static()
            value = np.arange(26).reshape(2, 13).astype(self.in_type_str)
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


support_types = get_xpu_op_support_types('adamw')
for stype in support_types:
    create_test_class(globals(), XPUTestAdamwOp1, stype)
    if stype == "float32":
        create_test_class(globals(), XPUTestAdamwOp2, stype)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
