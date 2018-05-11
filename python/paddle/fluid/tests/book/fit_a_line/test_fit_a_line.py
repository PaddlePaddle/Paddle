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

import paddle
import paddle.fluid as fluid
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


# train
def linear():
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    loss = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(loss)

    return avg_loss


def train(use_cuda, save_dirname, is_local):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    trainer = fluid.Trainer(
        linear, optimizer=fluid.optimizer.SGD(learning_rate=0.001), place=place)

    def event_handler(event):
        if isinstance(event, fluid.EndEpochEvent):
            test_metrics = trainer.test(
                reader=test_reader, feed_order=['x', 'y'])
            print test_metrics
            '''
            
            ...
            ['25.768919467926025']
            ['15.343549569447836']
            ...
            
            '''
            if float(test_metrics[0]) < 20.0:
                if save_dirname is not None:
                    # NOT clear yet
                    # fluid.io.save_inference_model(save_dirname, ['x'], [y_predict])
                    trainer.save_params(save_dirname)
            return

    trainer.train(
        reader=train_reader,
        num_epochs=100,
        event_handler=event_handler,
        feed_order=['x', 'y'])


# infer
def infer(use_cuda, save_dirname=None):
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
        tensor_x = numpy.random.uniform(0, 10,
                                        [batch_size, 13]).astype("float32")
        assert feed_target_names[0] == 'x'
        results = exe.run(inference_program,
                          feed={feed_target_names[0]: tensor_x},
                          fetch_list=fetch_targets)
        print("infer shape: ", results[0].shape)
        print("infer results: ", results[0])


def main(use_cuda, is_local=True):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    # Directory for saving the trained model
    save_dirname = "fit_a_line.inference.model"

    train(use_cuda, save_dirname, is_local)
    # inferencer.infer() NOT merged yet
    #infer(use_cuda, save_dirname)


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
