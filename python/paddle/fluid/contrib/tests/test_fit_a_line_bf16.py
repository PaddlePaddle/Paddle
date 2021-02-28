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
import contextlib
import numpy
import unittest

paddle.enable_static()


def train(is_local):
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

    with paddle.static.amp.bf16_guard():
        y_predict = fluid.layers.fc(input=x, size=1, act=None)

    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer = paddle.static.amp.decorate_bf16(
        sgd_optimizer, use_bf16_guard=True)
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
        sgd_optimizer.amp_init(exe.place)

        PASS_NUM = 1
        for pass_id in range(PASS_NUM):
            for data in train_reader():
                avg_loss_value, = exe.run(main_program,
                                          feed=feeder.feed(data),
                                          fetch_list=[avg_cost])
                print(avg_loss_value)

    if is_local:
        train_loop(fluid.default_main_program())


def infer(save_dirname=None):
    if save_dirname is None:
        return

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be fed
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)

        # The input's dimension should be 2-D and the second dim is 13
        # The input data should be >= 0
        batch_size = 10

        test_reader = paddle.batch(
            paddle.dataset.uci_housing.test(), batch_size=batch_size)

        test_data = next(test_reader())
        test_feat = numpy.array(
            [data[0] for data in test_data]).astype("float32")
        test_label = numpy.array(
            [data[1] for data in test_data]).astype("float32")

        assert feed_target_names[0] == 'x'
        results = exe.run(inference_program,
                          feed={feed_target_names[0]: numpy.array(test_feat)},
                          fetch_list=fetch_targets)
        print("infer shape: ", results[0].shape)
        print("infer results: ", results[0])
        print("ground truth: ", test_label)


def main():
    if not fluid.core.is_compiled_with_mkldnn():
        return

    # Directory for saving the trained model
    save_dirname = "fit_a_line.inference.model"

    train(save_dirname)
    infer(save_dirname)


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
