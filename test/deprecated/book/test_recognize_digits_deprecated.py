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

import math
import os
import sys
import unittest

import numpy

# TODO: remove sys.path.append
sys.path.append("../../legacy_test")
import nets

import paddle
from paddle import base
from paddle.base import core

paddle.enable_static()

BATCH_SIZE = 64


def loss_net(hidden, label):
    prediction = paddle.static.nn.fc(x=hidden, size=10, activation='softmax')
    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    avg_loss = paddle.mean(loss)
    acc = paddle.static.accuracy(input=prediction, label=label)
    return prediction, avg_loss, acc


def mlp(img, label):
    hidden = paddle.static.nn.fc(x=img, size=200, activation='tanh')
    hidden = paddle.static.nn.fc(x=hidden, size=200, activation='tanh')
    return loss_net(hidden, label)


def conv_net(img, label):
    conv_pool_1 = nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu",
    )
    conv_pool_1 = paddle.static.nn.batch_norm(conv_pool_1)
    conv_pool_2 = nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu",
    )
    return loss_net(conv_pool_2, label)


def train(
    nn_type,
    use_cuda,
    parallel,
    save_dirname=None,
    save_full_dirname=None,
    model_filename=None,
    params_filename=None,
    is_local=True,
):
    if use_cuda and not base.core.is_compiled_with_cuda():
        return
    img = paddle.static.data(name='img', shape=[-1, 1, 28, 28], dtype='float32')
    label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')

    if nn_type == 'mlp':
        net_conf = mlp
    else:
        net_conf = conv_net

    if parallel:
        raise NotImplementedError
    else:
        prediction, avg_loss, acc = net_conf(img, label)

    test_program = base.default_main_program().clone(for_test=True)

    optimizer = paddle.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()

    exe = base.Executor(place)

    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE,
    )
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=BATCH_SIZE
    )
    feeder = base.DataFeeder(feed_list=[img, label], place=place)

    def train_loop(main_program):
        exe.run(base.default_startup_program())

        PASS_NUM = 100
        for pass_id in range(PASS_NUM):
            for batch_id, data in enumerate(train_reader()):
                # train a mini-batch, fetch nothing
                exe.run(main_program, feed=feeder.feed(data))
                if (batch_id + 1) % 10 == 0:
                    acc_set = []
                    avg_loss_set = []
                    for test_data in test_reader():
                        acc_np, avg_loss_np = exe.run(
                            program=test_program,
                            feed=feeder.feed(test_data),
                            fetch_list=[acc, avg_loss],
                        )
                        acc_set.append(float(acc_np))
                        avg_loss_set.append(float(avg_loss_np))
                    # get test acc and loss
                    acc_val = numpy.array(acc_set).mean()
                    avg_loss_val = numpy.array(avg_loss_set).mean()
                    if float(acc_val) > 0.2 or pass_id == (PASS_NUM - 1):
                        # Smaller value to increase CI speed
                        if save_dirname is not None:
                            paddle.static.io.save_inference_model(
                                save_dirname,
                                img,
                                [prediction],
                                exe,
                            )
                        if save_full_dirname is not None:
                            paddle.static.save_inference_model(
                                save_full_dirname,
                                [],
                                [],
                                exe,
                            )
                        return
                    else:
                        print(
                            f'PassID {pass_id:1}, BatchID {batch_id + 1:04}, Test Loss {float(avg_loss_val):2.2}, Acc {float(acc_val):2.2}'
                        )
                        if math.isnan(float(avg_loss_val)):
                            sys.exit("got NaN loss, training failed.")
        raise AssertionError("Loss of recognize digits is too large")

    if is_local:
        train_loop(base.default_main_program())
    else:
        port = os.getenv("PADDLE_PSERVER_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVER_IPS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        trainers = int(os.getenv("PADDLE_TRAINERS"))
        current_endpoint = os.getenv("POD_IP") + ":" + port
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        training_role = os.getenv("PADDLE_TRAINING_ROLE", "TRAINER")
        t = paddle.distributed.transpiler.DistributeTranspiler()
        t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(
                current_endpoint, pserver_prog
            )
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            train_loop(t.get_trainer_program())


def infer(
    use_cuda, save_dirname=None, model_filename=None, params_filename=None
):
    if save_dirname is None:
        return

    place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
    exe = base.Executor(place)

    inference_scope = base.core.Scope()
    with base.scope_guard(inference_scope):
        # Use paddle.static.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be feeded
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [
            inference_program,
            feed_target_names,
            fetch_targets,
        ] = paddle.static.io.load_inference_model(
            save_dirname,
            exe,
        )

        # The input's dimension of conv should be 4-D or 5-D.
        # Use normilized image pixels as input data, which should be in the range [-1.0, 1.0].
        batch_size = 1
        tensor_img = numpy.random.uniform(
            -1.0, 1.0, [batch_size, 1, 28, 28]
        ).astype("float32")

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        results = exe.run(
            inference_program,
            feed={feed_target_names[0]: tensor_img},
            fetch_list=fetch_targets,
        )
        print("infer results: ", results[0])


def main(use_cuda, parallel, nn_type, combine):
    save_dirname = None
    save_full_dirname = None
    model_filename = None
    params_filename = None
    if not use_cuda and not parallel:
        save_dirname = "recognize_digits_" + nn_type + "_inference_model"
        save_full_dirname = "recognize_digits_" + nn_type + "_train_model"
        if combine:
            model_filename = "__model_combined__"
            params_filename = "__params_combined__"
            save_dirname = save_dirname + model_filename
            save_full_dirname = params_filename + params_filename

    # call train() with is_local argument to run distributed train
    train(
        nn_type=nn_type,
        use_cuda=use_cuda,
        parallel=parallel,
        save_dirname=save_dirname,
        save_full_dirname=save_full_dirname,
        model_filename=model_filename,
        params_filename=params_filename,
    )
    infer(
        use_cuda=use_cuda,
        save_dirname=save_dirname,
        model_filename=model_filename,
        params_filename=params_filename,
    )


class TestRecognizeDigits(unittest.TestCase):
    pass


def inject_test_method(use_cuda, parallel, nn_type, combine):
    def __impl__(self):
        prog = base.Program()
        startup_prog = base.Program()
        scope = base.core.Scope()
        with base.scope_guard(scope):
            with base.program_guard(prog, startup_prog):
                main(use_cuda, parallel, nn_type, combine)

    fn = 'test_{}_{}_{}_{}'.format(
        nn_type,
        'cuda' if use_cuda else 'cpu',
        'parallel' if parallel else 'normal',
        'combine' if combine else 'separate',
    )

    setattr(TestRecognizeDigits, fn, __impl__)


def inject_all_tests():
    for use_cuda in (False, True):
        if use_cuda and not core.is_compiled_with_cuda():
            continue
        for parallel in (False,):
            for nn_type in ('mlp', 'conv'):
                inject_test_method(use_cuda, parallel, nn_type, True)

    # Two unit-test for saving parameters as separate files
    inject_test_method(False, False, 'mlp', False)
    inject_test_method(False, False, 'conv', False)


inject_all_tests()

if __name__ == '__main__':
    unittest.main()
