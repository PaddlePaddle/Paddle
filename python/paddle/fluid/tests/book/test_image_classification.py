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

import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import contextlib
import math
import sys
import numpy as np
import unittest
import os


def conv_bn_layer(input, ch_out, filter_size, stride, padding, act='relu'):
    conv1 = fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv1, act=act)


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]
    if ch_in != ch_out:
        return conv_bn_layer(input, ch_out, 1, stride, 0, None)
    else:
        return input


def basicblock(input, ch_out, stride):
    short = shortcut(input, ch_out, stride)
    conv1 = conv_bn_layer(input, ch_out, 3, stride, 1)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1, act=None)
    return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')


def bottleneck(input, ch_out, stride):
    short = shortcut(input, ch_out * 4, stride)
    conv1 = conv_bn_layer(input, ch_out, 1, stride, 0)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1)
    conv3 = conv_bn_layer(conv2, ch_out * 4, 1, 1, 0, act=None)
    return fluid.layers.elementwise_add(x=short, y=conv3, act='relu')


def layer_warp(block_func, input, ch_out, count, stride):
    res_out = block_func(input, ch_out, stride)
    for i in range(1, count):
        res_out = block_func(res_out, ch_out, 1)
    return res_out


def resnet_imagenet(input, class_dim, depth=50, data_format='NCHW'):

    cfg = {
        18: ([2, 2, 2, 1], basicblock),
        34: ([3, 4, 6, 3], basicblock),
        50: ([3, 4, 6, 3], bottleneck),
        101: ([3, 4, 23, 3], bottleneck),
        152: ([3, 8, 36, 3], bottleneck)
    }
    stages, block_func = cfg[depth]
    conv1 = conv_bn_layer(input, ch_out=64, filter_size=7, stride=2, padding=3)
    pool1 = fluid.layers.pool2d(
        input=conv1, pool_type='avg', pool_size=3, pool_stride=2)
    res1 = layer_warp(block_func, pool1, 64, stages[0], 1)
    res2 = layer_warp(block_func, res1, 128, stages[1], 2)
    res3 = layer_warp(block_func, res2, 256, stages[2], 2)
    res4 = layer_warp(block_func, res3, 512, stages[3], 2)
    pool2 = fluid.layers.pool2d(
        input=res4,
        pool_size=7,
        pool_type='avg',
        pool_stride=1,
        global_pooling=True)
    out = fluid.layers.fc(input=pool2, size=class_dim, act='softmax')
    return out


def resnet_cifar10(input, class_dim, depth=32, data_format='NCHW'):
    assert (depth - 2) % 6 == 0

    n = (depth - 2) // 6

    conv1 = conv_bn_layer(
        input=input, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 64, n, 2)
    pool = fluid.layers.pool2d(
        input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    out = fluid.layers.fc(input=pool, size=class_dim, act='softmax')
    return out


def vgg16_bn_drop(input, class_dim):
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
    out = fluid.layers.fc(input=fc2, size=class_dim, act='softmax')
    return out


def train(net_type, data_set, use_cuda, save_dirname, is_local):
    if data_set == 'cifar10':
        class_dim = 10
        data_shape = [3, 32, 32]
    elif data_set == 'imagenet':
        class_dim = 102
        data_shape = [3, 224, 224]
    else:
        raise ValueError("%s data_set is not supported" % data_set)

    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if net_type == "vgg":
        print("train vgg on %s" % data_set)
        predict = vgg16_bn_drop(images, class_dim)
    elif net_type == "resnet":
        print("train resnet on %s" % data_set)
        if data_set == 'cifar10':
            predict = resnet_cifar10(images, class_dim)
        else:
            predict = resnet_imagenet(images, class_dim)
    else:
        raise ValueError("%s network is not supported" % net_type)

    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)

    acc = fluid.layers.accuracy(input=predict, label=label)

    # inference program
    #inference_program = fluid.default_main_program().clone()
    #with fluid.program_guard(inference_program):
    #    inference_program = fluid.io.get_inference_program(
    #                        target_vars=[batch_acc, batch_size_tensor])

    # test program
    test_program = fluid.default_main_program().clone(for_test=True)

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimize_ops, params_grads = optimizer.minimize(avg_cost)

    # fluid.memory_optimize(fluid.default_main_program())

    BATCH_SIZE = 128
    PASS_NUM = 100

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.cifar.train10()
            if data_set == 'cifar10' else paddle.dataset.flowers.train(),
            buf_size=128 * 10),
        batch_size=BATCH_SIZE)

    test_reader = paddle.batch(
        paddle.dataset.cifar.test10()
        if data_set == 'cifar10' else paddle.dataset.flowers.test(),
        batch_size=BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    #feeder = fluid.DataFeeder(place=place, feed_list=[images, label])

    def train_loop(main_program):
        exe.run(fluid.default_startup_program())
        loss = 0.0
        fluid.io.save_inference_model(save_dirname, ["pixel"], [predict], exe)
        """
        for pass_id in range(PASS_NUM):
            for batch_id, data in enumerate(train_reader()):
                train_image = np.array(
                    map(lambda x: x[0].reshape(data_shape), data)).astype(
                        "float32")
                train_label = np.array(map(lambda x: x[1], data)).astype(
                    "int64")
                train_label = train_label.reshape([-1, 1])

                # exe.run(main_program, feed=feeder.feed(data))
                exe.run(main_program,
                    feed={'pixel': train_image,
                          'label': train_label})

                if (batch_id % 10) == 0:
                    acc_list = []
                    avg_loss_list = []
                    for tid, test_data in enumerate(test_reader()):
                        test_image = np.array(
                            map(lambda x: x[0].reshape(data_shape),
                                test_data)).astype("float32")
                        test_label = np.array(map(lambda x: x[1], data)).astype(
                            "int64")
                        test_label = test_label.reshape([-1, 1])

                        loss_t, acc_t = exe.run(
                            program=test_program,
                            feed={"pixel": test_image,
                                  "label": test_label},
                            fetch_list=[avg_cost, acc])
                        if math.isnan(float(loss_t)):
                            sys.exit("got NaN loss, training failed.")
                        acc_list.append(float(acc_t))
                        avg_loss_list.append(float(loss_t))
                        break  # Use 1 segment for speeding up CI

                    acc_value = np.array(acc_list).mean()
                    avg_loss_value = np.array(avg_loss_list).mean()

                    print(
                        'PassID {0:1}, BatchID {1:04}, Test Loss {2:2.2}, Acc {3:2.2}'.
                        format(pass_id, batch_id + 1,
                               float(avg_loss_value), float(acc_value)))

                    if acc_value > 0.01:  # Low threshold for speeding up CI
                        fluid.io.save_inference_model(save_dirname, ["pixel"],
                                                      [predict], exe)
                        return
        """

    if is_local:
        train_loop(fluid.default_main_program())
    else:
        port = os.getenv("PADDLE_INIT_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_INIT_PSERVERS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        trainers = int(os.getenv("TRAINERS"))
        current_endpoint = os.getenv("POD_IP") + ":" + port
        trainer_id = int(os.getenv("PADDLE_INIT_TRAINER_ID"))
        training_role = os.getenv("TRAINING_ROLE", "TRAINER")
        t = fluid.DistributeTranspiler()
        t.transpile(
            optimize_ops,
            params_grads,
            trainer_id,
            pservers=pserver_endpoints,
            trainers=trainers)
        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            train_loop(t.get_trainer_program())


def infer(use_cuda, data_set, save_dirname=None):
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
        if data_set == 'cifar10':
            tensor_img = np.random.rand(batch_size, 3, 32, 32).astype("float32")
        else:
            tensor_img = np.random.rand(batch_size, 3, 224,
                                        224).astype("float32")

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        results = exe.run(inference_program,
                          feed={feed_target_names[0]: tensor_img},
                          fetch_list=fetch_targets)
        print("infer results: ", results[0])


def main(net_type, data_set, use_cuda, is_local=True):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    # Directory for saving the trained model
    save_dirname = "image_classification_" + net_type + "_" + data_set + ".inference.model"

    train(net_type, data_set, use_cuda, save_dirname, is_local)
    infer(use_cuda, data_set, save_dirname)


class TestImageClassification(unittest.TestCase):
    def test_vgg_imagenet_cuda(self):
        with self.scope_prog_guard():
            main('vgg', 'imagenet', use_cuda=True)

    #def test_resnet_imagenet_cuda(self):
    #    with self.scope_prog_guard():
    #        main('resnet', 'imagenet', use_cuda=True)

    #def test_vgg_cifar10_cuda(self):
    #    with self.scope_prog_guard():
    #        main('vgg', 'cifar10', use_cuda=True)

    #def test_resnet_cifar10_cuda(self):
    #    with self.scope_prog_guard():
    #        main('resnet', 'cifar10', use_cuda=True)

    #def test_vgg_cpu(self):
    #    with self.scope_prog_guard():
    #        main('vgg', use_cuda=False)

    #def test_resnet_cpu(self):
    #    with self.scope_prog_guard():
    #        main('resnet', use_cuda=False)

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
