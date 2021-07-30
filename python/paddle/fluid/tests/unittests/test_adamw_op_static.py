# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
from paddle.fluid import core
import paddle.fluid as fluid
import paddle


class testAdamWOpStatic(unittest.TestCase):
    def mlp(self, x, y):
        y_0 = fluid.layers.fc(input=x, size=10, act=None)
        y_0 = fluid.layers.dropout(y_0, dropout_prob=0.5)
        y_0 = fluid.layers.relu(y_0)
        y_1 = fluid.layers.fc(input=y_0, size=10, act=None)
        y_1 = fluid.layers.relu(y_1)
        y_2 = fluid.layers.fc(input=y_1, size=10, act=None)
        y_2 = fluid.layers.relu(y_2)
        y_pred = fluid.layers.fc(input=y_2, size=1, act=None)

        cost = fluid.layers.square_error_cost(input=y_pred, label=y)
        avg_cost = fluid.layers.mean(cost)

        return avg_cost

    def example_train_adamw(self, train_data, opt_type):
        paddle.seed(1234)
        paddle.enable_static()
        place = fluid.CUDAPlace(0)

        train_prog = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(train_prog, startup):
            x = fluid.data(name='x', shape=[None, 10], dtype='float32')
            y = fluid.data(name='y', shape=[None, 1], dtype='float32')

            avg_cost = self.mlp(x, y)

            if opt_type == "static":
                adam = fluid.optimizer.AdamWOptimizer(
                    learning_rate=0.01, weight_decay=0.01)
                adam.minimize(avg_cost)
            else:
                adam = paddle.optimizer.AdamW(
                    learning_rate=0.01, weight_decay=0.01)
                adam.minimize(avg_cost)

        exe = fluid.Executor(place)
        exe.run(startup)

        loss_list = []
        for data in train_data:
            loss_val = exe.run(train_prog,
                               feed={'x': data[0],
                                     'y': data[1]},
                               fetch_list=[avg_cost])
            loss_list.append(loss_val[0])

        paddle.disable_static()
        return loss_list

    def sample_train_data(self):
        np.random.seed(123)
        dataset = []
        for _ in range(30):
            train_data = []
            for _ in range(8):
                inputs = np.random.random(size=[10]).astype('float32')
                outputs = np.random.random(size=[1]).astype('float32')
                train_data.append([inputs, outputs])

            batch_size = len(train_data)
            x = np.array([d[0] for d in train_data]).reshape(batch_size, 10)
            y = np.array([d[1] for d in train_data]).reshape(batch_size, 1)
            dataset.append([x, y])

        return dataset

    def test_adamw_accuracy(self):
        train_data = self.sample_train_data()
        cost_static = self.example_train_adamw(train_data, "static")
        cost_dygraph = self.example_train_adamw(train_data, "dygraph")
        self.assertEqual((cost_static == cost_dygraph), True)

    def test_adamw_op_static(self):
        paddle.enable_static()
        place = fluid.CPUPlace()
        shape = [2, 3, 8, 8]
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

                opt = fluid.optimizer.AdamWOptimizer(
                    learning_rate=1e-5,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=0.01,
                    epsilon=1e-8)
                opt.minimize(loss)

        exe = fluid.Executor(place)
        exe.run(startup)
        data_np = np.random.random(shape).astype('float32')
        rets = exe.run(train_prog, feed={"data": data_np}, fetch_list=[loss])
        assert rets[0] is not None
        paddle.disable_static()

    def test_adamw_op_invalid_input(self):
        paddle.enable_static()
        train_prog = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(train_prog, startup):
            with fluid.unique_name.guard():
                with self.assertRaises(ValueError):
                    adamw = fluid.optimizer.AdamWOptimizer(
                        learning_rate=0.01, beta1=-1)
                with self.assertRaises(ValueError):
                    adamw = fluid.optimizer.AdamWOptimizer(
                        learning_rate=0.01, beta2=-1)
                with self.assertRaises(ValueError):
                    adamw = fluid.optimizer.AdamWOptimizer(
                        learning_rate=0.01, epsilon=-1)
                with self.assertRaises(ValueError):
                    adamw = fluid.optimizer.AdamWOptimizer(
                        learning_rate=0.01, weight_decay=-1)
                with self.assertRaises(TypeError):
                    adamw = fluid.optimizer.AdamWOptimizer(
                        learning_rate=0.01, weight_decay="0.01")

    def test_adamw_exception(self):
        paddle.enable_static()
        a = paddle.static.data(name="a", shape=[32, 32], dtype='float32')
        b = paddle.static.data(name="b", shape=[32, 32], dtype='float32')
        label = paddle.static.data(name="label", shape=[32, 1], dtype='int64')

        sum = paddle.add(a, b)
        z = paddle.pow(sum, 2.0)

        fc_1 = fluid.layers.fc(input=z, size=128)
        prediction = fluid.layers.fc(input=fc_1, size=2, act='softmax')

        cost = fluid.layers.cross_entropy(input=prediction, label=label)
        loss = fluid.layers.reduce_mean(cost)
        opt = fluid.optimizer.AdamWOptimizer(use_global_beta_pow=True)
        opt.minimize(loss)
        self.assertRaises(Exception, opt._get_global_accumulator, 'tmp')
        opt._add_global_accumulator('tmp', type=core.VarDesc.VarType.LOD_TENSOR)
        opt._get_global_accumulator('tmp')
        self.assertRaises(
            Exception,
            opt._add_global_accumulator,
            opt._beta1_pow_acc_str,
            type=core.VarDesc.VarType.LOD_TENSOR)
        paddle.disable_static()

    def test_adamw_op_coverage(self):
        paddle.enable_static()
        train_prog = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(train_prog, startup):
            with fluid.unique_name.guard():
                adamw = fluid.optimizer.AdamWOptimizer(
                    learning_rate=0.01, weight_decay=0.01)
                assert (adamw.__str__() is not None)


if __name__ == "__main__":
    unittest.main()
