# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import time
import numpy as np

os.environ["FLAGS_check_nan_inf"] = "1"

import paddle.fluid.core as core
import paddle
import paddle.fluid as fluid
import paddle.compat as cpt

np.random.seed(0)


def generator(nan_inf="nan"):
    for i in range(50):
        curr_train_x = np.random.randint(5, size=(5, 3)).astype("float32")
        if i >= 20:
            nan_inf = np.nan if nan_inf == "nan" else np.inf
            curr_train_x = np.ones(shape=(5, 3)).astype("float32") * nan_inf
        res = []
        for i in range(5):
            x = curr_train_x[i]
            y = x[0] + 2 * x[1] + 3 * x[2]
            res.append([y])
        y_true = np.array(res).astype('float32')
        yield [curr_train_x, y_true]


def net():
    x = fluid.layers.data(name="x", shape=[3], dtype='float32')
    y = fluid.layers.data(name="y", shape=[1], dtype='float32')
    hidden = x

    for i in range(2):
        hidden = fluid.layers.fc(input=hidden, size=6, act="sigmoid")

    y_predict = fluid.layers.fc(input=hidden, size=1, act=None)
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.05)
    sgd_optimizer.minimize(avg_cost)
    return y_predict, avg_cost


def check(use_cuda, nan_inf="nan"):
    main = fluid.Program()
    startup = fluid.Program()
    scope = fluid.core.Scope()

    with fluid.scope_guard(scope):
        with fluid.program_guard(main, startup):
            y_predict, avg_cost = net()

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup)

            step = 0.0
            for train_data, y_true in generator(nan_inf):
                outs = exe.run(main,
                               feed={'x': train_data,
                                     'y': y_true},
                               fetch_list=[y_predict.name, avg_cost.name])
                step += 1
                if step % 10 == 0:
                    print('iter={:.0f},cost={}'.format(step, outs[1][0]))


if __name__ == '__main__':
    if core.is_compiled_with_cuda():
        try:
            check(use_cuda=True)
            assert False
        except Exception as e:
            assert type(e) == core.EnforceNotMet
    try:
        check(use_cuda=False)
        assert False
    except Exception as e:
        assert type(e) == core.EnforceNotMet
