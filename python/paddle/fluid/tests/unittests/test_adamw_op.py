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
import paddle
import numpy as np
import paddle.fluid as fluid
from functools import partial


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
            weight_decay=0.01)

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
            weight_decay=0.01)
        assert (adam.__str__() is not None)

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

                beta1 = fluid.layers.create_global_var(
                    shape=[1], value=0.85, dtype='float32', persistable=True)
                beta2 = fluid.layers.create_global_var(
                    shape=[1], value=0.95, dtype='float32', persistable=True)
                betas = [beta1, beta2]
                opt = paddle.optimizer.AdamW(
                    learning_rate=1e-5,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=0.01,
                    epsilon=1e-8)
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
                0.1, beta1=-1, parameters=linear.parameters())
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.AdamW(
                0.1, beta2=-1, parameters=linear.parameters())
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.AdamW(
                0.1, epsilon=-1, parameters=linear.parameters())


class TestAdamWOpGroup(TestAdamWOp):
    def test_adamw_op_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        adam = paddle.optimizer.AdamW(
            learning_rate=0.01,
            parameters=[{
                'params': linear_1.parameters()
            }, {
                'params': linear_2.parameters(),
                'weight_decay': 0.001
            }],
            apply_decay_param_fun=lambda name: True,
            weight_decay=0.01)

        for _ in range(2):
            out = linear_1(a)
            out = linear_2(out)
            out.backward()
            adam.step()
            adam.clear_gradients()


class TestAdamWOpGroupWithLR(TestAdamWOp):
    def test_adamw_op_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        adam = paddle.optimizer.AdamW(
            learning_rate=paddle.optimizer.lr.PiecewiseDecay(
                boundaries=[3, 6], values=[0.1, 0.2, 0.3]),
            parameters=[{
                'params': linear_1.parameters(),
                'learning_rate': 0.1,
            }, {
                'params': linear_2.parameters(),
                'weight_decay': 0.001,
            }],
            apply_decay_param_fun=lambda name: True,
            weight_decay=0.01)

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

    return decay_rate**(n_layers + 2 - depth)


class TestAdamWOpLayerwiseLR(TestAdamWOp):
    def test_adamw_op_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear1 = paddle.nn.Linear(13, 8)
        linear2 = paddle.nn.Linear(8, 5)

        simple_lr_fun = partial(simple_lr_setting, decay_rate=0.8, n_layers=2)

        adam = paddle.optimizer.AdamW(
            learning_rate=0.01,
            parameters=[{
                'params': linear1.parameters()
            }, {
                'params': linear2.parameters(),
            }],
            apply_decay_param_fun=lambda name: True,
            weight_decay=0.01,
            lr_ratio=simple_lr_fun)

        for _ in range(2):
            a1 = linear1(a)
            out = linear2(a1)
            out.backward()
            adam.step()
            adam.clear_gradients()

    def test_adamw_op(self):
        paddle.enable_static()
        place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() \
            else fluid.CPUPlace()
        train_prog = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(train_prog, startup):
            with fluid.unique_name.guard():
                x = fluid.data(name='x', shape=[None, 10], dtype='float32')
                y = fluid.data(name='y', shape=[None, 1], dtype='float32')

                fc1 = fluid.layers.fc(input=x, size=32, act=None)
                prediction = fluid.layers.fc(input=fc1, size=1, act=None)
                cost = fluid.layers.square_error_cost(input=prediction, label=y)
                avg_cost = fluid.layers.mean(cost)

                simple_lr_fun = partial(
                    simple_lr_setting, decay_rate=0.8, n_layers=2)

                beta1 = fluid.layers.create_global_var(
                    shape=[1], value=0.85, dtype='float32', persistable=True)
                beta2 = fluid.layers.create_global_var(
                    shape=[1], value=0.95, dtype='float32', persistable=True)
                betas = [beta1, beta2]
                opt = paddle.optimizer.AdamW(
                    learning_rate=1e-5,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=0.01,
                    epsilon=1e-8,
                    lr_ratio=simple_lr_fun)
                opt.minimize(avg_cost)

        exe = fluid.Executor(place)
        exe.run(startup)
        for _ in range(2):
            inputs = np.random.random(size=[8, 10]).astype('float32')
            outputs = np.random.random(size=[8, 1]).astype('float32')
            rets = exe.run(train_prog,
                           feed={"x": inputs,
                                 "y": outputs},
                           fetch_list=[avg_cost])
            assert rets[0] is not None

        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
