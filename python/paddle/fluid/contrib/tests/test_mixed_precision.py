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
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)

        return x


class AMPTest(unittest.TestCase):
    def net():
        input_size = 4096
        output_size = 4096
        batch_size = 512
        train_data = [
            paddle.randn((batch_size, input_size)) for _ in range(nums_batch)
        ]
        labels = [
            paddle.randn((batch_size, output_size)) for _ in range(nums_batch)
        ]
        x = static.data(name='X', shape=[1000, 784], dtype='float32')
        y = static.data(name='Y', shape=[784, 100], dtype='float32')
        model = SimpleNet(input_size, output_size)  # 定义模型
        mse = paddle.nn.MSELoss()

        out = model(x)
        loss = mse(out, label)

        opt = paddle.fluid.optimizer.Adam(
            learning_rate=0.0001, parameters=model.parameters())  # 定义优化器
        opt = paddle.static.amp.decorate(
            opt, init_loss_scaling=128.0, use_dynamic_loss_scaling=True)
        opt.minimize(loss)
        return model, loss, opt

    def test_skip_update():
        with static.program_guard(startup_prog, main_prog):
            model, loss, opt = net()
        weight = model.linear1.weight
        moment1 = opt._optimizer._get_accumulator(
            self.opt._optimizer._moment1_acc_str, weight)
        beta_pow1 = opt._optimizer._get_accumulator(
            self.opt._optimizer._beta1_pow_acc_str, weight)
        fetch_list = [loss, weight, moment1, beta_pow1]
        exe = paddle.static.Executor(paddle.CUDAPlace(0))

        exe.run(startup_prog)
        for i in range(10):
            exe.run(main_prog, feed=feed, fetch_list=fetch_list)


if __name__ == '__main__':
    unittest.main()
