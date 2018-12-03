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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time
import os

import cProfile

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.profiler as profiler

import reader


def parse_args():
    parser = argparse.ArgumentParser('Convolution model benchmark.')
    parser.add_argument(
        '--model',
        type=str,
        choices=['resnet_imagenet', 'resnet_cifar10'],
        default='resnet_imagenet',
        help='The model architecture.')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='The minibatch size.')
    parser.add_argument(
        '--use_fake_data',
        action='store_true',
        help='use real data or fake data')
    parser.add_argument(
        '--skip_batch_num',
        type=int,
        default=0,
        help='The first num of minibatch num to skip, for better performance test.'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=0,
        help='The number of minibatches. 0 or less: whole dataset. Greater than 0: wraps the dataset up if necessary.'
    )
    parser.add_argument(
        '--pass_num', type=int, default=100, help='The number of passes.')
    parser.add_argument(
        '--data_format',
        type=str,
        default='NCHW',
        choices=['NCHW', 'NHWC'],
        help='The data data_format, now only support NCHW.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        '--data_set',
        type=str,
        default='flowers',
        choices=['cifar10', 'flowers', 'imagenet'],
        help='Optional dataset for benchmark.')
    parser.add_argument(
        '--infer_only', action='store_true', help='If set, run forward only.')
    parser.add_argument(
        '--use_cprof', action='store_true', help='If set, use cProfile.')
    parser.add_argument(
        '--use_nvprof',
        action='store_true',
        help='If set, use nvprof for CUDA.')
    parser.add_argument(
        '--skip_test',
        action='store_true',
        help='If set, skip testing the model during training.')
    parser.add_argument(
        '--save_model',
        action='store_true',
        help='If set, save the model after each epoch.')
    parser.add_argument(
        '--save_model_path',
        type=str,
        default='',
        help='A path for saving model.')
    parser.add_argument(
        '--train_file_list',
        type=str,
        default='data/ILSVRC2012/train_list.txt',
        help='A file with a list of training data files.')
    parser.add_argument(
        '--test_file_list',
        type=str,
        default='data/ILSVRC2012/val_list.txt',
        help='A file with a list of test data files.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/ILSVRC2012',
        help='A directory with train and test data files.')

    args = parser.parse_args()
    return args


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
    ch_in = input.shape[1] if args.data_format == 'NCHW' else input.shape[-1]
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
    out = fluid.layers.fc(input=pool2,
                          size=class_dim,
                          act='softmax',
                          bias_attr=False)
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
    out = fluid.layers.fc(input=pool,
                          size=class_dim,
                          act='softmax',
                          bias_attr=False)
    return out


def user_data_reader(data):
    """
    Creates a data reader whose data output is user data.
    """

    def data_reader():
        while True:
            for b in data:
                yield b

    return data_reader


def train(model, args):
    if args.use_cprof:
        pr = cProfile.Profile()
        pr.enable()

    if args.data_set == "cifar10":
        class_dim = 10
        if args.data_format == 'NCHW':
            dshape = [3, 32, 32]
        else:
            dshape = [32, 32, 3]
    elif args.data_set == "imagenet":
        class_dim = 1000
        if args.data_format == 'NCHW':
            #  dshape = [3, 256, 256]
            dshape = [3, 224, 224]
        else:
            #  dshape = [256, 256, 3]
            dshape = [224, 224, 3]
    else:
        class_dim = 102
        if args.data_format == 'NCHW':
            dshape = [3, 224, 224]
        else:
            dshape = [224, 224, 3]

    fake_train_data = [(
        np.random.rand(dshape[0] * dshape[1] * dshape[2]).astype(np.float32),
        np.random.randint(1, class_dim)) for _ in range(1)]
    fake_test_data = [(
        np.random.rand(dshape[0] * dshape[1] * dshape[2]).astype(np.float32),
        np.random.randint(1, class_dim)) for _ in range(1)]

    input = fluid.layers.data(name='data', shape=dshape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    predict = model(input, class_dim)
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
    batch_acc = fluid.layers.accuracy(
        input=predict, label=label, total=batch_size_tensor)

    inference_program = fluid.default_main_program().clone()
    """
    with fluid.program_guard(inference_program):
        inference_program = fluid.io.get_inference_program(
            target_vars=[batch_acc, batch_size_tensor])
    """
    # optimizer = fluid.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    opts = optimizer.minimize(avg_cost)

    fluid.memory_optimize(fluid.default_main_program())

    # Prepare fake data
    if args.use_fake_data:
        train_reader = paddle.batch(
            user_data_reader(fake_train_data), batch_size=args.batch_size)
        test_reader = paddle.batch(
            user_data_reader(fake_test_data), batch_size=args.batch_size)
    else:
        cycle = args.iterations > 0
        if args.data_set == 'cifar10':
            train_reader = paddle.batch(
                paddle.reader.shuffle(
                    paddle.dataset.cifar.train10(), buf_size=5120),
                batch_size=args.batch_size)
            test_reader = paddle.batch(
                paddle.dataset.cifar.test10(), batch_size=args.batch_size)
        elif args.data_set == 'imagenet':
            train_reader = paddle.batch(
                reader.train(
                    file_list=args.train_file_list,
                    data_dir=args.data_dir,
                    cycle=cycle),
                batch_size=args.batch_size)
            test_reader = paddle.batch(
                reader.test(
                    file_list=args.test_file_list,
                    data_dir=args.data_dir,
                    cycle=cycle),
                batch_size=args.batch_size)
        else:
            train_reader = paddle.batch(
                paddle.reader.shuffle(
                    paddle.dataset.flowers.train(), buf_size=5120),
                batch_size=args.batch_size)
            test_reader = paddle.batch(
                paddle.dataset.flowers.test(), batch_size=args.batch_size)

    def test(exe):
        test_accuracy = fluid.average.WeightedAverage()
        for batch_id, data in enumerate(test_reader()):
            img_data = np.array(map(lambda x: x[0].reshape(dshape),
                                    data)).astype("float32")
            y_data = np.array(map(lambda x: x[1], data)).astype("int64")
            y_data = y_data.reshape([-1, 1])

            acc, weight = exe.run(inference_program,
                                  feed={"data": img_data,
                                        "label": y_data},
                                  fetch_list=[batch_acc, batch_size_tensor])
            test_accuracy.add(value=acc, weight=weight)

        return test_accuracy.eval()

    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    accuracy = fluid.average.WeightedAverage()
    if args.use_fake_data:
        data = train_reader().next()
        image = np.array(map(lambda x: x[0].reshape(dshape), data)).astype(
            'float32')
        label = np.array(map(lambda x: x[1], data)).astype('int64')
        label = label.reshape([-1, 1])

    for pass_id in range(args.pass_num):
        accuracy.reset()
        train_accs = []
        train_losses = []
        batch_times = []
        fpses = []
        iters = 0
        total_samples = 0
        train_start_time = time.time()
        for data in train_reader():
            if args.iterations > 0 and iters == args.iterations + args.skip_batch_num:
                break
            if iters == args.skip_batch_num:
                profiler.reset_profiler()
                total_samples = 0
                train_start_time = time.time()
            if not args.use_fake_data:
                image = np.array(map(lambda x: x[0].reshape(dshape),
                                     data)).astype('float32')
                label = np.array(map(lambda x: x[1], data)).astype('int64')
                label = label.reshape([-1, 1])

            start = time.time()
            loss, acc, weight = exe.run(
                fluid.default_main_program(),
                feed={'data': image,
                      'label': label},
                fetch_list=[avg_cost, batch_acc, batch_size_tensor])
            batch_time = time.time() - start
            samples = len(label)
            total_samples += samples
            fps = samples / batch_time
            iters += 1
            accuracy.add(value=acc, weight=weight)
            train_losses.append(loss)
            train_accs.append(acc)
            batch_times.append(batch_time)
            fpses.append(fps)
            appx = ' (warm-up)' if iters <= args.skip_batch_num else ''
            print(
                "Pass: %d, Iter: %d%s, Loss: %f, Accuracy: %f, FPS: %.5f img/s"
                % (pass_id, iters, appx, loss, acc, fps))

        # Postprocess benchmark data
        latencies = batch_times[args.skip_batch_num:]
        latency_avg = np.average(latencies)
        latency_std = np.std(latencies)
        latency_pc99 = np.percentile(latencies, 99)
        fps_avg = np.average(fpses)
        fps_std = np.std(fpses)
        fps_pc01 = np.percentile(fpses, 1)
        train_total_time = time.time() - train_start_time
        examples_per_sec = total_samples / train_total_time

        # Benchmark output
        print("\nPass %d statistics:" % (pass_id))
        print("Loss: %f, Train Accuracy: %f" %
              (np.mean(train_losses), np.mean(train_accs)))
        print('Avg fps: %.5f, std fps: %.5f, fps for 99pc latency: %.5f' %
              (fps_avg, fps_std, fps_pc01))
        print('Avg latency: %.5f, std latency: %.5f, 99pc latency: %.5f' %
              (latency_avg, latency_std, latency_pc99))
        print('Total examples: %d, total time: %.5f, total examples/sec: %.5f\n'
              % (total_samples, train_total_time, examples_per_sec))

        # Save model
        if args.save_model:
            if not os.path.isdir(args.save_model_path):
                os.makedirs(args.save_model_path)
            fluid.io.save_inference_model(args.save_model_path,
                                          ["data", "label"],
                                          [batch_acc, batch_size_tensor], exe)
            print("Model saved into {}".format(args.save_model_path))

        # evaluation
        #  if not args.skip_test:
        #  pass_test_acc = test(exe)


def print_arguments(args):
    vars(args)['use_nvprof'] = (vars(args)['use_nvprof'] and
                                vars(args)['device'] == 'GPU')
    print('----------- resnet Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == '__main__':
    model_map = {
        'resnet_imagenet': resnet_imagenet,
        'resnet_cifar10': resnet_cifar10
    }
    args = parse_args()
    print_arguments(args)
    if args.data_format == 'NHWC':
        raise ValueError('Only support NCHW data_format now.')
    if args.use_nvprof and args.device == 'GPU':
        with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
            train(model_map[args.model], args)
    else:
        with profiler.profiler(args.device, sorted_key='total') as cpuprof:
            train(model_map[args.model], args)
