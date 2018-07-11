# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle.fluid as fluid
import paddle
import paddle.fluid.profiler as profiler

x = fluid.layers.data(name='x', shape=[10], dtype='float32')
y = fluid.layers.data(name='y', shape=[1], dtype='float32')
y_predict = fluid.layers.fc(input=x, size=1, act='softmax')
cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(cost)
# opt = fluid.optimizer.MixedPrecisionOptimizer(
#     scale_factor=128.0, learning_rate=0.001)
opt = fluid.optimizer.SGD(learning_rate=0.001)
opt = opt.minimize(avg_cost)

place = fluid.CPUPlace()
exe = fluid.Executor(place)

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=8192),
    batch_size=32)

exe.run(fluid.default_startup_program())
for pass_id in range(10):
    with profiler.profiler("All", 'total', '/tmp/profile') as pf:
        for data in train_reader():
            x_data = np.random.randn(32, 10).astype("float32")
            y_data = np.random.randint(
                low=0, high=2, size=(32, 1)).astype("float32")

            outs = exe.run(fluid.default_main_program(),
                           feed={'x': x_data,
                                 'y': y_data},
                           fetch_list=[avg_cost])
            print(outs[0])
