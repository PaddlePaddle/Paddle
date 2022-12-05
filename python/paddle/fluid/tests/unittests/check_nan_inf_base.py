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

import os

import numpy as np

os.environ[str("FLAGS_check_nan_inf")] = str("1")
os.environ[str("GLOG_vmodule")] = str("nan_inf_utils_detail=10")

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

paddle.enable_static()

np.random.seed(0)


def generator():
    batch_size = 5
    for i in range(5):
        curr_train_x = np.random.randint(
            batch_size, size=(batch_size, 3)
        ).astype("float32")
        if i >= 2:
            curr_train_x[0, :] = np.nan
            curr_train_x[-1, :] = np.inf
        res = []
        for i in range(batch_size):
            y = i % 3
            res.append([y])
        y_label = np.array(res).astype('int64')
        yield [curr_train_x, y_label]


def net():
    x = fluid.layers.data(name="x", shape=[3], dtype='float32')
    y = fluid.layers.data(name="y", shape=[1], dtype='int64')

    # test int64 value
    zero = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)

    # test float16 value
    fp16_zero = fluid.layers.cast(zero, dtype='float16')

    y = y + zero

    hidden = x

    for i in range(2):
        hidden = fluid.layers.fc(input=hidden, size=400, act="sigmoid")

    hidden = fluid.layers.fc(input=hidden, size=3, act=None)
    cost, y_predict = paddle.nn.functional.softmax_with_cross_entropy(
        hidden, y, return_softmax=True
    )
    acc_top1 = paddle.static.accuracy(input=y_predict, label=y, k=1)
    avg_cost = paddle.mean(cost)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.05)
    sgd_optimizer.minimize(avg_cost)
    return y_predict, avg_cost, acc_top1


def check(use_cuda):
    main = fluid.Program()
    startup = fluid.Program()
    scope = fluid.core.Scope()

    with fluid.scope_guard(scope):
        with fluid.program_guard(main, startup):
            y_predict, avg_cost, acc_top1 = net()

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup)

            step = 0.0
            for train_data, y_label in generator():
                outs = exe.run(
                    main,
                    feed={'x': train_data, 'y': y_label},
                    fetch_list=[y_predict.name, avg_cost.name, acc_top1.name],
                )
                step += 1
                print(
                    'iter={:.0f},cost={},acc1={}'.format(
                        step, outs[1][0], outs[2][0]
                    )
                )


if __name__ == '__main__':
    try:
        check(use_cuda=False)
        assert False
    except Exception as e:
        print(e)
        print(type(e))
        assert type(e) == RuntimeError

    if core.is_compiled_with_cuda():
        try:
            check(use_cuda=True)
            assert False
        except Exception as e:
            print(e)
            print(type(e))
            # Note. Enforce in cuda kernel may not catch in paddle, and
            # Exception type will be RuntimeError
            assert type(e) == OSError or type(e) == RuntimeError
