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
import contextlib
import math
import sys
import numpy as np
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


def resnet(input, class_dim, depth=50, data_format='NCHW'):
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
    return pool2


def vgg16(input, class_dim):
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


def train(net_type, data_set, place, save_dirname, threshold=0.005):
    class_dim = 102
    data_shape = [3, 224, 224]

    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if net_type == "vgg":
        print("train vgg")
        net = vgg16(images, class_dim)
    elif net_type == "resnet":
        print("train resnet")
        net = resnet(images, class_dim)
    else:
        raise ValueError("%s network is not supported" % net_type)

    predict = fluid.layers.fc(input=net, size=class_dim, act='softmax')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=predict, label=label)

    #Test program
    test_program = fluid.default_main_program().clone(for_test=True)
    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_cost)

    BATCH_SIZE = 32
    PASS_NUM = 100

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.flowers.train(), buf_size=128 * 10),
        batch_size=BATCH_SIZE)

    test_reader = paddle.batch(
        paddle.dataset.flowers.test(), batch_size=BATCH_SIZE)

    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(place=place, feed_list=[images, label])

    exe.run(fluid.default_startup_program())
    main_program = fluid.default_main_program()

    for pass_id in range(PASS_NUM):
        for batch_id, data in enumerate(train_reader()):
            train_image = np.array(
                map(lambda x: x[0].reshape(data_shape), data)).astype("float32")
            train_label = np.array(map(lambda x: x[1], data)).astype("int64")
            train_label = train_label.reshape([-1, 1])

            exe.run(main_program,
                    feed={'pixel': train_image,
                          'label': train_label})

            if (batch_id % 100) == 0:
                acc_list = []
                avg_loss_list = []
                for tid, test_data in enumerate(test_reader()):
                    test_image = np.array(
                        map(lambda x: x[0].reshape(data_shape),
                            test_data)).astype("float32")
                    test_label = np.array(map(lambda x: x[1],
                                              test_data)).astype("int64")
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


def test_accuracy(data_set, executor, inference_program, feed_target_names,
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


def infer(data_set, place, save_dirname):
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

        #print("The test set accuracy of inference in float mode is:")
        #test_accuracy(data_set, exe, inference_program, feed_target_names, fetch_targets)

        float16_inference_program = inference_program.clone()
        t = fluid.InferenceTranspiler()
        t.float16_transpile(float16_inference_program, place)

        fp16_save_dirname = "float16_" + save_dirname
        fluid.io.save_inference_model(fp16_save_dirname, feed_target_names,
                                      fetch_targets, exe,
                                      float16_inference_program)

        #print("The test set accuracy of inference in float16 mode is:")
        #test_accuracy(exe, float16_inference_program, feed_target_names, fetch_targets)


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

    net_types = ["vgg"]
    data_sets = ["imagenet"]
    for data_set in data_sets:
        for net in net_types:
            with scope_prog_guard():
                save_dirname = "image_classification_" + data_set + "_" + net + ".inference.model"
                train(net, data_set, place, save_dirname)
                infer(data_set, place, save_dirname)
