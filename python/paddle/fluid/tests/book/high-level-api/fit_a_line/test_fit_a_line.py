#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import sys
try:
    from paddle.fluid.contrib.trainer import *
    from paddle.fluid.contrib.inferencer import *
except ImportError:
    print(
        "In the fluid 1.0, the trainer and inferencer are moving to paddle.fluid.contrib",
        file=sys.stderr)
    from paddle.fluid.trainer import *
    from paddle.fluid.inferencer import *
import contextlib
import numpy
import unittest

# train reader
BATCH_SIZE = 20

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.uci_housing.train(), buf_size=500),
    batch_size=BATCH_SIZE)

test_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.uci_housing.test(), buf_size=500),
    batch_size=BATCH_SIZE)


def inference_program():
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    return y_predict


def train_program():
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    y_predict = inference_program()

    loss = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(loss)

    return [avg_loss, y_predict]


def optimizer_func():
    return fluid.optimizer.SGD(learning_rate=0.001)


def train(use_cuda, train_program, params_dirname, inference_model_dirname):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    trainer = Trainer(
        train_func=train_program, place=place, optimizer_func=optimizer_func)

    def event_handler(event):
        if isinstance(event, EndStepEvent):
            if event.step == 10:
                test_metrics = trainer.test(
                    reader=test_reader, feed_order=['x', 'y'])
                print(test_metrics)
                '''
                ...
                ['25.768919467926025']
                ['15.343549569447836']
                ...
                '''
                if params_dirname is not None:
                    trainer.save_params(params_dirname)
                    trainer.save_inference_model(inference_model_dirname,
                                                 ['x'], [1])
                trainer.stop()

    trainer.train(
        reader=train_reader,
        num_epochs=100,
        event_handler=event_handler,
        feed_order=['x', 'y'])


# infer
def infer(use_cuda, inference_program, params_dirname=None):
    if params_dirname is None:
        return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    inferencer = Inferencer(
        infer_func=inference_program, param_path=params_dirname, place=place)

    batch_size = 10
    tensor_x = numpy.random.uniform(0, 10, [batch_size, 13]).astype("float32")

    results = inferencer.infer({'x': tensor_x})
    print("infer results: ", results[0])


def infer_by_saved_model(use_cuda, save_dirname=None):
    if save_dirname is None:
        return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be feeded
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


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    # Directory for saving the trained model
    params_dirname = "fit_a_line.model"
    inference_model_dirname = "fit_a_line.inference_model"

    train(use_cuda, train_program, params_dirname, inference_model_dirname)
    infer(use_cuda, inference_program, params_dirname)
    infer_by_saved_model(use_cuda, inference_model_dirname)


class TestFitALine(unittest.TestCase):
    def test_cpu(self):
        with self.program_scope_guard():
            with fluid.unique_name.guard():
                main(use_cuda=False)

    def test_cuda(self):
        with self.program_scope_guard():
            with fluid.unique_name.guard():
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
