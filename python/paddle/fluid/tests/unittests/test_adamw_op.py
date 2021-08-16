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

    def test_adamw_lr_decay(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)

        lr = paddle.optimizer.lr.NoamDecay(d_model=0.01, warmup_steps=10)
        wd = 0.1
        adam = paddle.optimizer.AdamW(
            learning_rate=lr,
            parameters=linear.parameters(),
            apply_decay_param_fun=lambda name: True,
            weight_decay=wd)

        for _ in range(2):
            out = linear(a)
            out.backward()
            lr_to_coeff = adam._lr_to_coeff
            adam.step()

            for i, value in enumerate(lr_to_coeff.values()):
                self.assertAlmostEqual(value.numpy()[0], 1.0 - lr() * wd)
            self.assertEqual(len(adam._lr_to_coeff), 0)

            lr.step()
            adam.clear_gradients()


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


if __name__ == "__main__":
    unittest.main()
