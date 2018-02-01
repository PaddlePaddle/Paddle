#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import contextlib
import unittest


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    x = fluid.layers.data(name='x', shape=[13], dtype='float32')

    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(x=cost)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_cost)

    BATCH_SIZE = 20

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.uci_housing.train(), buf_size=500),
        batch_size=BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())

    PASS_NUM = 100
    for pass_id in range(PASS_NUM):
        fluid.io.save_persistables(exe, "./fit_a_line.model/")
        fluid.io.load_persistables(exe, "./fit_a_line.model/")
        for data in train_reader():
            avg_loss_value, = exe.run(fluid.default_main_program(),
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost])
            print(avg_loss_value)
            if avg_loss_value[0] < 10.0:
                return
    raise AssertionError("Fit a line cost is too large, {0:2.2}".format(
        avg_loss_value[0]))


class TestFitALine(unittest.TestCase):
    def test_cpu(self):
        with self.program_scope_guard():
            main(use_cuda=False)

    def test_cuda(self):
        with self.program_scope_guard():
            main(use_cuda=True)

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
