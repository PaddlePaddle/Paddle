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

import argparse
import paddle
import paddle.fluid as fluid
import contextlib
import math
import sys
import unittest
import os
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
    n = (depth - 2) / 6
    conv1 = conv_bn_layer(
        input=input, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 32, 64, n, 2)
    pool = fluid.layers.pool2d(
        input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    return pool


def train(place, save_dirname, threshold):
    classdim = 10
    data_shape = [3, 32, 32]

    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    print("Start training resnet")
    net = resnet_cifar10(images, 32)

    predict = fluid.layers.fc(input=net, size=classdim, act='softmax')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=predict, label=label)

    #Test program
    test_program = fluid.default_main_program().clone(for_test=True)

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_cost)

    BATCH_SIZE = 128
    PASS_NUM = 100

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.cifar.train10(), buf_size=128 * 10),
        batch_size=BATCH_SIZE)

    test_reader = paddle.batch(
        paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE)

    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(place=place, feed_list=[images, label])

    exe.run(fluid.default_startup_program())
    main_program = fluid.default_main_program()

    for pass_id in range(PASS_NUM):
        for batch_id, data in enumerate(train_reader()):
            exe.run(main_program, feed=feeder.feed(data))

            if (batch_id % 100) == 0:
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

                acc_value = np.array(acc_list).mean()
                avg_loss_value = np.array(avg_loss_list).mean()

                print(
                    'PassID {0:1}, BatchID {1:04}, Test Loss {2:2.2}, Accuracy {3:2.2}'.
                    format(pass_id, batch_id + 1,
                           float(avg_loss_value), float(acc_value)))

                if acc_value > threshold:
                    print(
                        'Save inference model with test accuracy of {0} at {1}'.
                        format(float(acc_value), save_dirname))
                    fluid.io.save_inference_model(save_dirname, ["pixel"],
                                                  [predict], exe)
                    return


def test_accuracy(executor, inference_program, feed_target_names,
                  fetch_targets):
    test_reader = paddle.dataset.cifar.test10()
    test_num = 0
    batch_size = 100
    correct_num = 0
    imgs = []
    labels = []

    for item in test_reader():
        label = item[1]
        img = item[0].astype(np.float32)
        imgs.append(img.reshape(3, 32, 32))
        labels.append(label)
        if len(imgs) == batch_size:
            batch_imgs = np.stack(imgs, axis=0)
            results = executor.run(inference_program,
                                   feed={feed_target_names[0]: batch_imgs},
                                   fetch_list=fetch_targets)
            prediction = np.argmax(results[0], axis=1)
            correct_num += np.sum(prediction == labels)
            test_num += batch_size
            imgs = []
            labels = []

    print("{0} out of {1} predictions are correct.".format(correct_num,
                                                           test_num))
    print("Test accuray is {0}.".format(float(correct_num) / float(test_num)))


def test_accuracy_new(executor, inference_program, feed_target_names,
                      fetch_targets):
    batch_size = 128
    test_reader = paddle.batch(
        paddle.dataset.cifar.test10(), batch_size=batch_size)
    data_shape = [3, 32, 32]
    test_num = 0
    correct_num = 0

    for test_data in test_reader():
        test_image = np.array(
            map(lambda x: x[0].reshape(data_shape), test_data)).astype(
                "float32")
        test_label = np.array(map(lambda x: x[1], test_data)).astype("int64")
        test_label = test_label.reshape([-1, 1])

        results = executor.run(program=inference_program,
                               feed={feed_target_names[0]: test_image},
                               fetch_list=fetch_targets)

        prediction = np.argmax(results[0], axis=1).reshape([-1, 1])
        correct_num += np.sum(prediction == test_label)
        test_num += test_label.size

    print("{0} out of {1} predictions are correct.".format(correct_num,
                                                           test_num))
    print("New Test accuray is {0}.".format(
        float(correct_num) / float(test_num)))


def infer(place, save_dirname):
    exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be feeded
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        print("Load inference model from {0}".format(save_dirname))
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)

        print("The test set accuracy of inference in float mode is:")
        test_accuracy(exe, inference_program, feed_target_names, fetch_targets)
        test_accuracy_new(exe, inference_program, feed_target_names,
                          fetch_targets)

        float16_inference_program = inference_program.clone()
        t = fluid.InferenceTranspiler()
        t.float16_transpile(float16_inference_program, place)

        print("The test set accuracy of inference in float16 mode is:")
        test_accuracy(exe, float16_inference_program, feed_target_names,
                      fetch_targets)
        test_accuracy_new(exe, float16_inference_program, feed_target_names,
                          fetch_targets)


@contextlib.contextmanager
def scope_prog_guard():
    prog = fluid.Program()
    startup_prog = fluid.Program()
    scope = fluid.core.Scope()
    with fluid.scope_guard(scope):
        with fluid.program_guard(prog, startup_prog):
            yield


if __name__ == "__main__":
    if not fluid.core.is_compiled_with_cuda():
        raise Exception("This test requires CUDA GPUs!")

    place = fluid.CUDAPlace(0)
    if not fluid.core.is_float16_supported(place):
        raise Exception(
            "This test requires compute capability of CUDA GPU >= 5.3!")

    save_dirname = "image_classification_resnet.inference.model"

    parser = argparse.ArgumentParser(
        description='float16 inference accuracy test')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.6,
        help='Save inference model when test accuracy reach this threshold.')
    parser.add_argument(
        '--repeat', type=int, default=1, help='The number of tests to run.')
    args = parser.parse_args()

    print("This inference accuracy experiment will repeat test {0} times.".
          format(args.repeat))
    for i in range(args.repeat):
        with scope_prog_guard():
            print("\n\nTest #{0}:\n".format(i + 1))
            train(place, save_dirname, args.threshold)
            infer(place, save_dirname)
