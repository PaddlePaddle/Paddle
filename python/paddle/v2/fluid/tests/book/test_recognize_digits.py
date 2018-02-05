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
from __future__ import print_function
import argparse
import paddle.v2.fluid as fluid
import paddle.v2 as paddle
import sys
import numpy
import unittest


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "nn_type",
        help="The neural network type, in ['mlp', 'conv']",
        type=str,
        choices=['mlp', 'conv'])
    parser.add_argument(
        "--parallel",
        help='Run in parallel or not',
        default=False,
        action="store_true")
    parser.add_argument(
        "--use_cuda",
        help="Run the program by using CUDA",
        default=False,
        action="store_true")
    return parser.parse_args()


BATCH_SIZE = 64


def loss_net(hidden, label):
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(x=loss)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return prediction, avg_loss, acc


def mlp(img, label):
    hidden = fluid.layers.fc(input=img, size=200, act='tanh')
    hidden = fluid.layers.fc(input=hidden, size=200, act='tanh')
    return loss_net(hidden, label)


def conv_net(img, label):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    return loss_net(conv_pool_2, label)


def train(nn_type, use_cuda, parallel, save_dirname):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if nn_type == 'mlp':
        net_conf = mlp
    else:
        net_conf = conv_net

    if parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            img_ = pd.read_input(img)
            label_ = pd.read_input(label)
            prediction, avg_loss, acc = net_conf(img_, label_)
            for o in [avg_loss, acc]:
                pd.write_output(o)

        avg_loss, acc = pd()
        # get mean loss and acc through every devices.
        avg_loss = fluid.layers.mean(x=avg_loss)
        acc = fluid.layers.mean(x=acc)
    else:
        prediction, avg_loss, acc = net_conf(img, label)

    test_program = fluid.default_main_program().clone()

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    PASS_NUM = 100
    for pass_id in range(PASS_NUM):
        for batch_id, data in enumerate(train_reader()):
            # train a mini-batch, fetch nothing
            exe.run(feed=feeder.feed(data))
            if (batch_id + 1) % 10 == 0:
                acc_set = []
                avg_loss_set = []
                for test_data in test_reader():
                    acc_np, avg_loss_np = exe.run(program=test_program,
                                                  feed=feeder.feed(test_data),
                                                  fetch_list=[acc, avg_loss])
                    acc_set.append(float(acc_np))
                    avg_loss_set.append(float(avg_loss_np))
                # get test acc and loss
                acc_val = numpy.array(acc_set).mean()
                avg_loss_val = numpy.array(avg_loss_set).mean()
                if float(acc_val) > 0.85:  # test acc > 85%
                    if save_dirname is not None:
                        fluid.io.save_inference_model(save_dirname, ["img"],
                                                      [prediction], exe)
                    return
                else:
                    print(
                        'PassID {0:1}, BatchID {1:04}, Test Loss {2:2.2}, Acc {3:2.2}'.
                        format(pass_id, batch_id + 1,
                               float(avg_loss_val), float(acc_val)))
    raise AssertionError("Loss of recognize digits is too large")


def infer(use_cuda, save_dirname=None):
    if save_dirname is None:
        return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # Use fluid.io.load_inference_model to obtain the inference program desc,
    # the feed_target_names (the names of variables that will be feeded 
    # data using feed operators), and the fetch_targets (variables that 
    # we want to obtain data from using fetch operators).
    [inference_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)

    # The input's dimension of conv should be 4-D or 5-D.
    tensor_img = numpy.random.rand(1, 1, 28, 28).astype("float32")

    # Construct feed as a dictionary of {feed_target_name: feed_target_data}
    # and results will contain a list of data corresponding to fetch_targets.
    results = exe.run(inference_program,
                      feed={feed_target_names[0]: tensor_img},
                      fetch_list=fetch_targets)
    print("infer results: ", results[0])


def main(use_cuda, parallel, nn_type):
    if not use_cuda and not parallel:
        save_dirname = "recognize_digits_" + nn_type + ".inference.model"
    else:
        save_dirname = None

    train(
        nn_type=nn_type,
        use_cuda=use_cuda,
        parallel=parallel,
        save_dirname=save_dirname)
    infer(use_cuda=use_cuda, save_dirname=save_dirname)


class TestRecognizeDigits(unittest.TestCase):
    pass


def inject_test_method(use_cuda, parallel, nn_type):
    def __impl__(self):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                main(use_cuda, parallel, nn_type)

    fn = 'test_{0}_{1}_{2}'.format(nn_type, 'cuda'
                                   if use_cuda else 'cpu', 'parallel'
                                   if parallel else 'normal')

    setattr(TestRecognizeDigits, fn, __impl__)


def inject_all_tests():
    for use_cuda in (False, True):
        for parallel in (False, True):
            for nn_type in ('mlp', 'conv'):
                inject_test_method(use_cuda, parallel, nn_type)


inject_all_tests()

if __name__ == '__main__':
    unittest.main()
