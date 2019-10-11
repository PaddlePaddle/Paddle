#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import math
import sys
import numpy
import unittest
import os
import copy
import numpy as np


def resnet_cifar10(input, depth=32):
    def conv_bn_layer(input,
                      ch_out,
                      filter_size,
                      stride,
                      padding,
                      act='relu',
                      bias_attr=False):
        tmp = fluid.layers.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            bias_attr=bias_attr)
        return fluid.layers.batch_norm(input=tmp, act=act)

    def shortcut(input, ch_in, ch_out, stride):
        if ch_in != ch_out:
            return conv_bn_layer(input, ch_out, 1, stride, 0, None)
        else:
            return input

    def basicblock(input, ch_in, ch_out, stride):
        tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
        tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
        short = shortcut(input, ch_in, ch_out, stride)
        return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')

    def layer_warp(block_func, input, ch_in, ch_out, count, stride):
        tmp = block_func(input, ch_in, ch_out, stride)
        for i in range(1, count):
            tmp = block_func(tmp, ch_out, ch_out, 1)
        return tmp

    assert (depth - 2) % 6 == 0
    n = (depth - 2) // 6
    conv1 = conv_bn_layer(
        input=input, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 32, 64, n, 2)
    pool = fluid.layers.pool2d(
        input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    return pool


def vgg16_bn_drop(input):
    def conv_block(input, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=input,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=4096, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=4096, act=None)
    return fc2


def train(net_type, use_cuda, save_dirname, is_local):
    classdim = 10
    data_shape = [3, 32, 32]

    train_program = fluid.Program()
    startup_prog = fluid.Program()
    train_program.random_seed = 123
    startup_prog.random_seed = 456
    with fluid.program_guard(train_program, startup_prog):
        images = fluid.layers.data(
            name='pixel', shape=data_shape, dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        if net_type == "vgg":
            print("train vgg net")
            net = vgg16_bn_drop(images)
        elif net_type == "resnet":
            print("train resnet")
            net = resnet_cifar10(images, 32)
        else:
            raise ValueError("%s network is not supported" % net_type)

        logits = fluid.layers.fc(input=net, size=classdim, act="softmax")
        cost, predict = fluid.layers.softmax_with_cross_entropy(
            logits, label, return_softmax=True)
        avg_cost = fluid.layers.mean(cost)
        acc = fluid.layers.accuracy(input=predict, label=label)

        # Test program
        test_program = train_program.clone(for_test=True)

        optimizer = fluid.optimizer.Lamb(learning_rate=0.001)

        mp_optimizer = fluid.contrib.mixed_precision.decorate(
            optimizer=optimizer,
            init_loss_scaling=8.0,
            use_dynamic_loss_scaling=True)

        mp_optimizer.minimize(avg_cost)
        scaled_loss = mp_optimizer.get_loss_scaling()

    BATCH_SIZE = 128
    PASS_NUM = 1

    # no shuffle for unit test
    train_reader = paddle.batch(
        paddle.dataset.cifar.train10(), batch_size=BATCH_SIZE)

    test_reader = paddle.batch(
        paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(place=place, feed_list=[images, label])

    def train_loop(main_program):
        exe.run(startup_prog)
        loss = 0.0
        for pass_id in range(PASS_NUM):
            for batch_id, data in enumerate(train_reader()):
                np_scaled_loss, loss = exe.run(
                    main_program,
                    feed=feeder.feed(data),
                    fetch_list=[scaled_loss, avg_cost])
                print(
                    'PassID {0:1}, BatchID {1:04}, train loss {2:2.4}, scaled train closs {3:2.4}'.
                    format(pass_id, batch_id + 1,
                           float(loss), float(np_scaled_loss)))
                if (batch_id % 10) == 0:
                    acc_list = []
                    avg_loss_list = []
                    for tid, test_data in enumerate(test_reader()):
                        loss_t, acc_t = exe.run(program=test_program,
                                                feed=feeder.feed(test_data),
                                                fetch_list=[avg_cost, acc])
                        if math.isnan(float(loss_t)):
                            sys.exit("got NaN loss, training failed.")
                        acc_list.append(float(acc_t))
                        avg_loss_list.append(float(loss_t))
                        break  # Use 1 segment for speeding up CI

                    acc_value = numpy.array(acc_list).mean()
                    avg_loss_value = numpy.array(avg_loss_list).mean()

                    print(
                        'PassID {0:1}, BatchID {1:04}, test loss {2:2.2}, acc {3:2.2}'.
                        format(pass_id, batch_id + 1,
                               float(avg_loss_value), float(acc_value)))

                    if acc_value > 0.08:  # Low threshold for speeding up CI
                        fluid.io.save_inference_model(
                            save_dirname, ["pixel"], [predict],
                            exe,
                            main_program=train_program)
                        return

    if is_local:
        train_loop(train_program)
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
        t = fluid.DistributeTranspiler()
        t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            train_loop(t.get_trainer_program())


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

        # The input's dimension of conv should be 4-D or 5-D.
        # Use normilized image pixels as input data, which should be in the range [0, 1.0].
        batch_size = 1
        tensor_img = numpy.random.rand(batch_size, 3, 32, 32).astype("float32")

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        results = exe.run(inference_program,
                          feed={feed_target_names[0]: tensor_img},
                          fetch_list=fetch_targets)

        print("infer results: ", results[0])

        fluid.io.save_inference_model(save_dirname, feed_target_names,
                                      fetch_targets, exe, inference_program)


def main(net_type, use_cuda, is_local=True):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    # Directory for saving the trained model
    save_dirname = "image_classification_" + net_type + ".inference.model"

    train(net_type, use_cuda, save_dirname, is_local)
    #infer(use_cuda, save_dirname)


class TestImageClassification(unittest.TestCase):
    def test_amp_lists(self):
        white_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.white_list)
        black_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.black_list)
        gray_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.gray_list)

        amp_lists = fluid.contrib.mixed_precision.AutoMixedPrecisionLists()
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_1(self):
        white_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.white_list)
        black_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.black_list)
        gray_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.gray_list)

        # 1. w={'exp}, b=None
        white_list.add('exp')
        black_list.remove('exp')

        amp_lists = fluid.contrib.mixed_precision.AutoMixedPrecisionLists(
            {'exp'})
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_2(self):
        white_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.white_list)
        black_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.black_list)
        gray_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.gray_list)

        # 2. w={'tanh'}, b=None
        white_list.add('tanh')
        gray_list.remove('tanh')

        amp_lists = fluid.contrib.mixed_precision.AutoMixedPrecisionLists(
            {'tanh'})
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_3(self):
        white_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.white_list)
        black_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.black_list)
        gray_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.gray_list)

        # 3. w={'lstm'}, b=None
        white_list.add('lstm')

        amp_lists = fluid.contrib.mixed_precision.AutoMixedPrecisionLists(
            {'lstm'})
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_4(self):
        white_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.white_list)
        black_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.black_list)
        gray_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.gray_list)

        # 4. w=None, b={'conv2d'}
        white_list.remove('conv2d')
        black_list.add('conv2d')

        amp_lists = fluid.contrib.mixed_precision.AutoMixedPrecisionLists(
            custom_black_list={'conv2d'})
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_5(self):
        white_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.white_list)
        black_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.black_list)
        gray_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.gray_list)

        # 5. w=None, b={'tanh'}
        black_list.add('tanh')
        gray_list.remove('tanh')

        amp_lists = fluid.contrib.mixed_precision.AutoMixedPrecisionLists(
            custom_black_list={'tanh'})
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_6(self):
        white_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.white_list)
        black_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.black_list)
        gray_list = copy.copy(
            fluid.contrib.mixed_precision.fp16_lists.gray_list)

        # 6. w=None, b={'lstm'}
        black_list.add('lstm')

        amp_lists = fluid.contrib.mixed_precision.AutoMixedPrecisionLists(
            custom_black_list={'lstm'})
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_7(self):
        # 7. w={'lstm'} b={'lstm'}
        # raise ValueError
        self.assertRaises(ValueError,
                          fluid.contrib.mixed_precision.AutoMixedPrecisionLists,
                          {'lstm'}, {'lstm'})

    def test_vgg_cuda(self):
        with self.scope_prog_guard():
            main('vgg', use_cuda=True)

    def test_resnet_cuda(self):
        with self.scope_prog_guard():
            main('resnet', use_cuda=True)

    @contextlib.contextmanager
    def scope_prog_guard(self):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                yield


if __name__ == '__main__':
    unittest.main()
