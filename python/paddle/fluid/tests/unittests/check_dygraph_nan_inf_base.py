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

from __future__ import unicode_literals
from __future__ import print_function

import os
import sys
import time
import numpy as np

os.environ[str("FLAGS_check_nan_inf")] = str("1")

import paddle.fluid.core as core
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
from paddle.fluid.dygraph.base import to_variable

np.random.seed(0)


def generator():
    batch_size = 5
    for i in range(5):
        curr_train_x = np.random.randint(
            batch_size, size=(batch_size, 3)).astype("float32")
        if i >= 2:
            curr_train_x[0, :] = np.nan
            curr_train_x[-1, :] = np.inf
        res = []
        for i in range(batch_size):
            y = i % 3
            res.append([y])
        y_label = np.array(res).astype('int64')
        yield [curr_train_x, y_label]


class Net(fluid.dygraph.Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.fc0 = Linear(3, 400, act="sigmoid")
        self.fc1 = Linear(400, 400, act="sigmoid")
        self.fc2 = Linear(400, 3, act=None)

    def forward(self, x, y):
        # test int64 value
        #        zero = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
        # test float16 value
        #        fp16_zero = fluid.layers.cast(zero, dtype='float16')
        #        y = y + zero

        hidden = x
        hidden = self.fc0(hidden)
        hidden = self.fc1(hidden)
        hidden = self.fc2(hidden)
        cost, y_predict = fluid.layers.softmax_with_cross_entropy(
            hidden, y, return_softmax=True)
        acc_top1 = fluid.layers.accuracy(input=y_predict, label=y, k=1)
        avg_cost = fluid.layers.mean(cost)
        return avg_cost


def check(use_cuda):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = Net()
        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.05,
                                            parameter_list=model.parameters())
        model.train()
        step = 0
        for train_data, y_label in generator():
            x = to_variable(train_data)
            y = to_variable(y_label)
            y.stop_gradient = True
            avg_cost = model(x, y)
            print("-------------------finish forward--------------------")
            avg_cost.backward()
            sgd_optimizer.minimize(avg_cost)
            model.clear_gradients()
            step += 1
            print('iter={:.0f},cost={}'.format(step, avg_cost))


if __name__ == '__main__':
    if core.is_compiled_with_cuda():
        try:
            check(use_cuda=True)
            assert False
        except Exception as e:
            print(e)
            print(type(e))
            # Note. Enforce in cuda kernel may not catch in paddle, and
            # Exception type will be RuntimeError
            assert type(e) == core.EnforceNotMet or type(e) == RuntimeError
#    try:
#        check(use_cuda=False)
#        assert False
#    except Exception as e:
#        print(e)
#        print(type(e))
#        assert type(e) == core.EnforceNotMet
