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

DATA_TYPE = np.float16
BATCH_SIZE = 5
fluid.default_startup_program().random_seed = 100
np.random.seed(100)

x = fluid.layers.data(name='x', shape=[1, 28, 28], dtype=DATA_TYPE)
y = fluid.layers.data(name='y', shape=[1], dtype=DATA_TYPE)
fc = fluid.layers.fc(input=x, size=10, act='relu')
y_predict = fluid.layers.fc(input=fc, size=1, act='relu')
cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(cost)
opt = fluid.optimizer.MixedSGD(scale_factor=1.0, learning_rate=0.001)
# opt = fluid.optimizer.SGD(learning_rate=0.001)
opt = opt.minimize(avg_cost)

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place, log_level=10)

train_reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=BATCH_SIZE)

exe.run(fluid.default_startup_program())
for pass_id in range(10):
    with profiler.profiler("All", 'total', '/tmp/profile') as pf:
        for i, data in enumerate(train_reader()):
            # x_data = np.random.uniform(low=-1.0, high=1.0, size=(BATCH_SIZE, 10)).astype(DATA_TYPE)
            # y_data = np.random.randint(
            #     low=0, high=2, size=(BATCH_SIZE, 1)).astype(DATA_TYPE)

            x_data = np.array(map(lambda x: x[0].reshape([1, 28, 28]),
                                  data)).astype(DATA_TYPE)
            y_data = np.array(map(lambda x: x[1], data)).astype(DATA_TYPE)
            y_data = y_data.reshape([len(y_data), 1])

            # print(fluid.default_main_program())
            # exit(0)
            outs = exe.run(fluid.default_main_program(),
                           feed={'x': x_data,
                                 'y': y_data},
                           fetch_list=[avg_cost])
            print("pass {0}, batch {1}, loss {2}".format(pass_id, i, outs[0]))
            if i == 0:
                exit(0)
