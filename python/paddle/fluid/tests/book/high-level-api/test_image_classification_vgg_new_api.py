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

import sys

import paddle
import paddle.fluid as fluid

try:
    from paddle.fluid.contrib.trainer import *
    from paddle.fluid.contrib.inferencer import *
except ImportError:
    print(
        "In the fluid 1.0, the trainer and inferencer are moving to paddle.fluid.contrib",
        file=sys.stderr)
    from paddle.fluid.trainer import *
    from paddle.fluid.inferencer import *
import paddle.fluid.core as core
import numpy
import os
import cifar10_small_test_set


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
    predict = fluid.layers.fc(input=fc2, size=10, act='softmax')
    return predict


def inference_network():
    data_shape = [3, 32, 32]
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    predict = vgg16_bn_drop(images)
    return predict


def train_network():
    predict = inference_network()
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, accuracy]


def optimizer_func():
    return fluid.optimizer.Adam(learning_rate=0.001)


def train(use_cuda, train_program, parallel, params_dirname):
    BATCH_SIZE = 128
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            cifar10_small_test_set.train10(batch_size=10), buf_size=128 * 10),
        batch_size=BATCH_SIZE,
        drop_last=False)
    # Use only part of the test set data validation program
    test_reader = paddle.batch(
        cifar10_small_test_set.test10(BATCH_SIZE),
        batch_size=BATCH_SIZE,
        drop_last=False)

    def event_handler(event):
        if isinstance(event, EndStepEvent):
            avg_cost, accuracy = trainer.test(
                reader=test_reader, feed_order=['pixel', 'label'])

            print('Loss {0:2.2}, Acc {1:2.2}'.format(avg_cost, accuracy))

            if accuracy > 0.01:  # Low threshold for speeding up CI
                if params_dirname is not None:
                    trainer.save_params(params_dirname)
                return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    trainer = Trainer(
        train_func=train_program,
        place=place,
        optimizer_func=optimizer_func,
        parallel=parallel)

    trainer.train(
        reader=train_reader,
        num_epochs=1,
        event_handler=event_handler,
        feed_order=['pixel', 'label'])


def infer(use_cuda, inference_program, parallel, params_dirname=None):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    inferencer = Inferencer(
        infer_func=inference_program,
        param_path=params_dirname,
        place=place,
        parallel=parallel)

    # The input's dimension of conv should be 4-D or 5-D.
    # Use normilized image pixels as input data, which should be in the range
    # [0, 1.0].
    tensor_img = numpy.random.rand(1, 3, 32, 32).astype("float32")
    results = inferencer.infer({'pixel': tensor_img})

    print("infer results: ", results)


def main(use_cuda, parallel):
    save_path = "image_classification_vgg.inference.model"

    os.environ['CPU_NUM'] = str(4)
    train(
        use_cuda=use_cuda,
        train_program=train_network,
        params_dirname=save_path,
        parallel=parallel)

    # FIXME(zcd): in the inference stage, the number of
    # input data is one, it is not appropriate to use parallel.
    if parallel and use_cuda:
        return
    os.environ['CPU_NUM'] = str(1)
    infer(
        use_cuda=use_cuda,
        inference_program=inference_network,
        params_dirname=save_path,
        parallel=parallel)


if __name__ == '__main__':
    for use_cuda in (False, True):
        for parallel in (False, True):
            if use_cuda and not core.is_compiled_with_cuda():
                continue
            main(use_cuda=use_cuda, parallel=parallel)
