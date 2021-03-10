#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import paddle.static.amp as amp
import contextlib
import unittest
import math
import sys

paddle.enable_static()


def train():
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

    with paddle.static.amp.bf16_guard():
        y_predict = fluid.layers.fc(input=x, size=1, act=None)

    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    amp.rewrite_program_bf16(fluid.default_main_program(), use_bf16_guard=True)
    sgd_optimizer.minimize(avg_cost)

    BATCH_SIZE = 20

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.uci_housing.train(), buf_size=500),
        batch_size=BATCH_SIZE)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    def train_loop(main_program):
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe.run(fluid.default_startup_program())

        PASS_NUM = 100
        for pass_id in range(PASS_NUM):
            for data in train_reader():
                avg_loss_value, = exe.run(main_program,
                                          feed=feeder.feed(data),
                                          fetch_list=[avg_cost])
                print(avg_loss_value)
                if avg_loss_value[0] < 10.0:
                    return
                if math.isnan(float(avg_loss_value)):
                    sys.exit("got NaN loss, training failed.")
        raise AssertionError("Fit a line cost is too large, {0:2.2}".format(
            avg_loss_value[0]))

    train_loop(fluid.default_main_program())


def main():
    if not fluid.core.is_compiled_with_mkldnn():
        return

    train()


class TestFitALine(unittest.TestCase):
    def test_cpu(self):
        with self.program_scope_guard():
            main()

    @contextlib.contextmanager
    def program_scope_guard(self):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                yield


if __name__ == '__main__':
    unittest.main()
