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
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.contrib.mixed_precision import fp16_utils
import paddle
import paddle.nn as nn
import paddle.static as static
import numpy as np

paddle.enable_static()


class SimpleNet(nn.Layer):
    def __init__(self, input_size, output_size):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_size, output_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_size, output_size)

    def forward(self, x):

        x = self.linear1(x)
        # currently, paddle's relu may hide nan/inf, relu(nan) = 0, relu(inf)= inf
        # so, do not use it here.
        #x = self.relu1(x) 
        x = self.linear2(x)
        #x = self.relu2(x)
        x = self.linear3(x)

        return x


class AMPTest(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)

    def net(self):
        input_size = 4096
        output_size = 4096
        x = static.data(name='X', shape=[1000, 4096], dtype='float32')
        label = static.data(name='Y', shape=[1000, 4096], dtype='float32')
        model = SimpleNet(input_size, output_size)  # 定义模型
        mse = paddle.nn.MSELoss()

        out = model(x)
        loss = mse(out, label)

        opt = paddle.fluid.optimizer.Adam(
            learning_rate=0.0001, parameter_list=model.parameters())  # 定义优化器
        opt = paddle.static.amp.decorate(
            opt, init_loss_scaling=128.0, use_dynamic_loss_scaling=True)
        opt.minimize(loss)
        return model, loss, opt

    def test_skip_update(self):
        input_size = 4096
        output_size = 4096
        batch_size = 1000
        nums_batch = 10
        startup_prog = paddle.static.Program()
        main_prog = paddle.static.Program()
        with static.program_guard(main_prog, startup_prog):
            model, loss, opt = self.net()
            weight = model.linear1.weight
            moment1 = opt._optimizer._get_accumulator(
                opt._optimizer._moment1_acc_str, weight)
            beta_pow1 = opt._optimizer._get_accumulator(
                opt._optimizer._beta1_pow_acc_str, weight)
            fetch_list = [
                loss, weight, moment1, beta_pow1, 'find_infinite_scale.tmp_0'
            ]

            exe = paddle.static.Executor(self.place)

            train_data = [
                np.random.rand(batch_size, input_size).astype(np.float32)
                for _ in range(nums_batch)
            ]
            labels = [
                np.random.rand(batch_size, output_size).astype(np.float32)
                for _ in range(nums_batch)
            ]

            weight_, moment1_, beta_pow1_ = exe.run(
                startup_prog, fetch_list=[weight, moment1, beta_pow1])
            pre_weight_, pre_moment1_, pre_beta_pow1_ = weight_, moment1_, beta_pow1_
            for i in range(nums_batch):
                if i % 2:
                    train_data[i][10] = np.inf
                loss_, weight_, moment1_, beta_pow1_, found_inf = exe.run(
                    main_prog,
                    feed={"X": train_data[i],
                          "Y": labels[i]},
                    fetch_list=fetch_list)
                print(loss_, weight_[0][0], moment1_[0][0], beta_pow1_,
                      found_inf)
                if i % 2:
                    self.assertTrue(found_inf)
                    self.assertTrue(np.array_equal(weight_, pre_weight_))
                    self.assertTrue(np.array_equal(moment1_, pre_moment1_))
                    self.assertTrue(np.array_equal(beta_pow1_, pre_beta_pow1_))
                else:
                    self.assertFalse(found_inf)
                    self.assertFalse(np.array_equal(weight_, pre_weight_))
                    self.assertFalse(np.array_equal(moment1_, pre_moment1_))
                    self.assertFalse(np.array_equal(beta_pow1_, pre_beta_pow1_))
                pre_weight_, pre_moment1_, pre_beta_pow1_ = weight_, moment1_, beta_pow1_


if __name__ == '__main__':
    unittest.main()
