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

import contextlib
import math
import os
import sys
import tempfile
import unittest

import numpy

# TODO: remove sys.path.append
sys.path.append("../../legacy_test")
import nets

import paddle
from paddle import base

paddle.enable_static()


def resnet_cifar10(input, depth=32):
    def conv_bn_layer(
        input, ch_out, filter_size, stride, padding, act='relu', bias_attr=False
    ):
        tmp = paddle.static.nn.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            bias_attr=bias_attr,
        )
        return paddle.static.nn.batch_norm(input=tmp, act=act)

    def shortcut(input, ch_in, ch_out, stride):
        if ch_in != ch_out:
            return conv_bn_layer(input, ch_out, 1, stride, 0, None)
        else:
            return input

    def basicblock(input, ch_in, ch_out, stride):
        tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
        tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
        short = shortcut(input, ch_in, ch_out, stride)
        return paddle.nn.functional.relu(paddle.add(x=tmp, y=short))

    def layer_warp(block_func, input, ch_in, ch_out, count, stride):
        tmp = block_func(input, ch_in, ch_out, stride)
        for i in range(1, count):
            tmp = block_func(tmp, ch_out, ch_out, 1)
        return tmp

    assert (depth - 2) % 6 == 0
    n = (depth - 2) // 6
    conv1 = conv_bn_layer(
        input=input, ch_out=16, filter_size=3, stride=1, padding=1
    )
    res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 32, 64, n, 2)
    pool = paddle.nn.functional.avg_pool2d(x=res3, kernel_size=8, stride=1)
    return pool


def vgg16_bn_drop(input):
    def conv_block(input, num_filter, groups, dropouts):
        return nets.img_conv_group(
            input=input,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max',
        )

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = paddle.nn.functional.dropout(x=conv5, p=0.5)
    fc1 = paddle.static.nn.fc(x=drop, size=4096)
    bn = paddle.static.nn.batch_norm(input=fc1, act='relu')
    drop2 = paddle.nn.functional.dropout(x=bn, p=0.5)
    fc2 = paddle.static.nn.fc(x=drop2, size=4096)
    return fc2


def train(net_type, use_cuda, save_dirname, is_local):
    classdim = 10
    data_shape = [3, 32, 32]

    images = paddle.static.data(
        name='pixel', shape=[-1, *data_shape], dtype='float32'
    )
    label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')

    if net_type == "vgg":
        print("train vgg net")
        net = vgg16_bn_drop(images)
    elif net_type == "resnet":
        print("train resnet")
        net = resnet_cifar10(images, 32)
    else:
        raise ValueError(f"{net_type} network is not supported")

    predict = paddle.static.nn.fc(x=net, size=classdim, activation='softmax')
    cost = paddle.nn.functional.cross_entropy(
        input=predict, label=label, reduction='none', use_softmax=False
    )
    avg_cost = paddle.mean(cost)
    acc = paddle.static.accuracy(input=predict, label=label)

    # Test program
    test_program = base.default_main_program().clone(for_test=True)

    optimizer = paddle.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_cost)

    BATCH_SIZE = 128
    PASS_NUM = 1

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.cifar.train10(), buf_size=128 * 10
        ),
        batch_size=BATCH_SIZE,
    )

    test_reader = paddle.batch(
        paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE
    )

    place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
    exe = base.Executor(place)
    feeder = base.DataFeeder(place=place, feed_list=[images, label])

    def train_loop(main_program):
        exe.run(base.default_startup_program())
        loss = 0.0
        for pass_id in range(PASS_NUM):
            for batch_id, data in enumerate(train_reader()):
                exe.run(main_program, feed=feeder.feed(data))

                if (batch_id % 10) == 0:
                    acc_list = []
                    avg_loss_list = []
                    for tid, test_data in enumerate(test_reader()):
                        loss_t, acc_t = exe.run(
                            program=test_program,
                            feed=feeder.feed(test_data),
                            fetch_list=[avg_cost, acc],
                        )
                        if math.isnan(float(loss_t)):
                            sys.exit("got NaN loss, training failed.")
                        acc_list.append(float(acc_t))
                        avg_loss_list.append(float(loss_t))
                        break  # Use 1 segment for speeding up CI

                    acc_value = numpy.array(acc_list).mean()
                    avg_loss_value = numpy.array(avg_loss_list).mean()

                    print(
                        f'PassID {pass_id:1}, BatchID {batch_id + 1:04}, Test Loss {float(avg_loss_value):2.2}, Acc {float(acc_value):2.2}'
                    )

                    if acc_value > 0.01:  # Low threshold for speeding up CI
                        paddle.static.io.save_inference_model(
                            save_dirname, images, [predict], exe
                        )
                        return

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


def infer(use_cuda, save_dirname=None):
    if save_dirname is None:
        return

    place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
    exe = base.Executor(place)

    inference_scope = base.core.Scope()
    with base.scope_guard(inference_scope):
        # Use paddle.static.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be fed
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [
            inference_program,
            feed_target_names,
            fetch_targets,
        ] = paddle.static.io.load_inference_model(save_dirname, exe)

        # The input's dimension of conv should be 4-D or 5-D.
        # Use normilized image pixels as input data, which should be in the range [0, 1.0].
        batch_size = 1
        tensor_img = numpy.random.rand(batch_size, 3, 32, 32).astype("float32")

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        results = exe.run(
            inference_program,
            feed={feed_target_names[0]: tensor_img},
            fetch_list=fetch_targets,
        )

        print("infer results: ", results[0])
        feeded_vars = [
            inference_program.global_block().var(name)
            for name in feed_target_names
        ]
        paddle.static.io.save_inference_model(
            save_dirname,
            feeded_vars,
            fetch_targets,
            exe,
            program=inference_program,
        )


def main(net_type, use_cuda, is_local=True):
    if use_cuda and not base.core.is_compiled_with_cuda():
        return

    # Directory for saving the trained model
    temp_dir = tempfile.TemporaryDirectory()
    save_dirname = os.path.join(
        temp_dir.name, "image_classification_" + net_type + "_inference_model"
    )

    train(net_type, use_cuda, save_dirname, is_local)
    infer(use_cuda, save_dirname)
    temp_dir.cleanup()


class TestImageClassification(unittest.TestCase):
    def test_vgg_cuda(self):
        with self.scope_prog_guard():
            main('vgg', use_cuda=True)

    def test_resnet_cuda(self):
        with self.scope_prog_guard():
            main('resnet', use_cuda=True)

    def test_vgg_cpu(self):
        with self.scope_prog_guard():
            main('vgg', use_cuda=False)

    def test_resnet_cpu(self):
        with self.scope_prog_guard():
            main('resnet', use_cuda=False)

    @contextlib.contextmanager
    def scope_prog_guard(self):
        prog = base.Program()
        startup_prog = base.Program()
        scope = base.core.Scope()
        with base.scope_guard(scope):
            with base.program_guard(prog, startup_prog):
                yield


if __name__ == '__main__':
    unittest.main()
