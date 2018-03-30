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
import os
import numpy as np
import sys
import time
import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import argparse
import functools

batch_size = 16

train_reader = paddle.batch(paddle.dataset.flowers.train(), batch_size)


def trans_to_recordio(reader, dst_file_name):
    feeder = fluid.DataFeeder(
        feed_list=[
            fluid.layers.data(
                name='image', shape=[3, 224, 224], dtype='float32'),
            fluid.layers.data(
                name='label', shape=[1], dtype='int64')
        ],
        place=fluid.CPUPlace())
    fluid.recordio_writer.convert_reader_to_recordio_file(
        dst_file_name,
        reader,
        feeder,
        compressor=core.RecordIOWriter.Compressor.NoCompress)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--batch_size',
    type=int,
    default=batch_size,
    help="Batch size for training.")
parser.add_argument(
    '--learning_rate',
    type=float,
    default=1e-3,
    help="Learning rate for training.")
parser.add_argument('--num_passes', type=int, default=1, help="No. of passes.")
parser.add_argument(
    '--device',
    type=str,
    default='GPU',
    choices=['CPU', 'GPU'],
    help="The device type.")
parser.add_argument(
    '--data_format',
    type=str,
    default='NCHW',
    choices=['NCHW', 'NHWC'],
    help='The data order, now only support NCHW.')
parser.add_argument(
    '--data_set',
    type=str,
    default='flowers',
    choices=['cifar10', 'flowers'],
    help='Optional dataset for benchmark.')
args = parser.parse_args()


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
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    return fc2


def main_train():
    if args.data_set == "cifar10":
        classdim = 10
        if args.data_format == 'NCHW':
            data_shape = [3, 32, 32]
        else:
            data_shape = [32, 32, 3]
    else:
        classdim = 102
        if args.data_format == 'NCHW':
            data_shape = [3, 224, 224]
        else:
            data_shape = [224, 224, 3]

    # Input data
    """
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    """
    data_file = fluid.layers.open_recordio_file(
        filename='./flowers_16.recordio',
        shapes=[[-1, 3, 224, 224], [-1, 1]],
        lod_levels=[0, 0],
        dtypes=['float32', 'int64'])
    data_file = fluid.layers.create_double_buffer_reader(
        reader=data_file, place='CUDA:0')
    images, label = fluid.layers.read_file(data_file)

    # Train program
    net = vgg16_bn_drop(images)
    predict = fluid.layers.fc(input=net, size=classdim, act='softmax')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    # Evaluator
    batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
    batch_acc = fluid.layers.accuracy(
        input=predict, label=label, total=batch_size_tensor)

    # inference program
    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        inference_program = fluid.io.get_inference_program(
            target_vars=[batch_acc, batch_size_tensor])

    # Optimization
    optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
    opts = optimizer.minimize(avg_cost)

    fluid.memory_optimize(fluid.default_main_program())

    # Initialize executor
    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)
    exe = fluid.Executor(place)

    # Parameter initialization
    exe.run(fluid.default_startup_program())

    # data reader
    """
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.cifar.train10()
            if args.data_set == 'cifar10' else paddle.dataset.flowers.train(),
            buf_size=5120),
        batch_size=args.batch_size)
    """
    test_reader = paddle.batch(
        paddle.dataset.cifar.test10()
        if args.data_set == 'cifar10' else paddle.dataset.flowers.test(),
        batch_size=args.batch_size)

    # test
    def test(exe):
        test_accuracy = fluid.average.WeightedAverage()
        for batch_id, data in enumerate(test_reader()):
            img_data = np.array(map(lambda x: x[0].reshape(data_shape),
                                    data)).astype("float32")
            y_data = np.array(map(lambda x: x[1], data)).astype("int64")
            y_data = y_data.reshape([-1, 1])

            acc, weight = exe.run(inference_program,
                                  feed={"pixel": img_data,
                                        "label": y_data},
                                  fetch_list=[batch_acc, batch_size_tensor])
            test_accuracy.add(value=acc, weight=weight)
        return test_accuracy.eval()

    iters = 0
    accuracy = fluid.average.WeightedAverage()

    # here
    #print(fluid.default_main_program())

    for pass_id in range(args.num_passes):
        # train
        start_time = time.time()
        num_samples = 0
        accuracy.reset()
        while not data_file.eof():

            loss, acc, weight = exe.run(
                fluid.default_main_program(),
                fetch_list=[avg_cost, batch_acc, batch_size_tensor])
            accuracy.add(value=acc, weight=weight)
            iters += 1
            num_samples += args.batch_size
            print(
                "Pass = %d, Iters = %d, Loss = %f, Accuracy = %f" %
                (pass_id, iters, loss, acc)
            )  # The accuracy is the accumulation of batches, but not the current batch.

        pass_elapsed = time.time() - start_time
        pass_train_acc = accuracy.eval()
        # pass_test_acc = test(exe)
        print(
            "Pass = %d, Training performance = %f imgs/s, Train accuracy = %f\n"
            % (pass_id, num_samples / pass_elapsed, pass_train_acc))


def print_arguments():
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    print("Trans to recodio...")
    trans_to_recordio(train_reader, './flowers_16.recordio')
    print("Complete trans.")
    print_arguments()
    main_train()
