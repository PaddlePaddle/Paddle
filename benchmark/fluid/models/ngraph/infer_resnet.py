# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler

import reader


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def parse_args():
    parser = argparse.ArgumentParser('Convolution model benchmark.')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='The minibatch size.')
    parser.add_argument(
        '--use_fake_data',
        action='store_true',
        help='If set, use fake data instead of real data.')
    parser.add_argument(
        '--skip_batch_num',
        type=int,
        default=0,
        help='The number of the first minibatches to skip in statistics, for better performance test.'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=0,
        help='The number of minibatches to process. 0 or less: whole dataset. Greater than 0: cycle the dataset if needed.'
    )
    parser.add_argument(
        '--data_format',
        type=str,
        default='NCHW',
        choices=['NCHW', 'NHWC'],
        help='The data data_format, now only support NCHW.')
    parser.add_argument(
        '--device',
        type=str,
        default='CPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        '--data_set',
        type=str,
        default='flowers',
        choices=['cifar10', 'flowers', 'imagenet'],
        help='Optional dataset for benchmark.')
    parser.add_argument(
        '--profile', action='store_true', help='If set, do profiling.')
    parser.add_argument(
        '--infer_model_path',
        type=str,
        default='',
        help='The directory for loading inference model.')
    parser.add_argument(
        '--test_file_list',
        type=str,
        default='data/ILSVRC2012/val_list.txt',
        help='A file with a list of test data files.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/ILSVRC2012',
        help='A directory with test data files.')

    args = parser.parse_args()
    return args


def user_data_reader(data):
    '''
    Creates a data reader for user data.
    '''

    def data_reader():
        while True:
            for b in data:
                yield b

    return data_reader


def infer(args):
    if not os.path.exists(args.infer_model_path):
        raise IOError("Invalid inference model path!")

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

    fake_data = [(
        np.random.rand(dshape[0] * dshape[1] * dshape[2]).astype(np.float32),
        np.random.randint(1, class_dim)) for _ in range(1)]

    image = fluid.layers.data(name='data', shape=dshape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    place = fluid.CUDAPlace(0) if args.device == 'GPU' else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # load model
    [infer_program, feed_dict,
     fetch_targets] = fluid.io.load_inference_model(args.infer_model_path, exe)

    # infer data read
    if args.use_fake_data:
        infer_reader = paddle.batch(
            user_data_reader(fake_data), batch_size=args.batch_size)
    else:
        cycle = args.iterations > 0
        if args.data_set == 'cifar10':
            infer_reader = paddle.batch(
                paddle.dataset.cifar.test10(cycle=cycle),
                batch_size=args.batch_size)
        elif args.data_set == 'imagenet':
            infer_reader = paddle.batch(
                reader.test(
                    file_list=args.test_file_list,
                    data_dir=args.data_dir,
                    cycle=cycle),
                batch_size=args.batch_size)
        else:
            infer_reader = paddle.batch(
                paddle.dataset.flowers.test(cycle=cycle),
                batch_size=args.batch_size)

    infer_accuracy = fluid.metrics.Accuracy()

    if args.use_fake_data:
        data = infer_reader().next()
        image = np.array(map(lambda x: x[0].reshape(dshape), data)).astype(
            'float32')
        label = np.array(map(lambda x: x[1], data)).astype('int64')
        label = label.reshape([-1, 1])

    infer_accs = []
    iters = 0
    batch_times = []
    fpses = []
    total_samples = 0
    infer_start_time = time.time()
    for data in infer_reader():
        if args.iterations > 0 and iters == args.iterations + args.skip_batch_num:
            break
        if iters == args.skip_batch_num:
            profiler.reset_profiler()
            total_samples = 0
            infer_start_time = time.time()
        if not args.use_fake_data:
            image = np.array(map(lambda x: x[0].reshape(dshape), data)).astype(
                "float32")
            label = np.array(map(lambda x: x[1], data)).astype("int64")
            label = label.reshape([-1, 1])

        start = time.time()
        acc, weight = exe.run(infer_program,
                              feed={feed_dict[0]: image,
                                    feed_dict[1]: label},
                              fetch_list=fetch_targets)

        batch_time = time.time() - start
        samples = len(label)
        total_samples += samples
        fps = samples / batch_time
        batch_times.append(batch_time)
        fpses.append(fps)
        infer_accuracy.update(value=acc, weight=weight)
        infer_acc = infer_accuracy.eval()
        infer_accs.append(infer_acc)
        iters += 1
        appx = ' (warm-up)' if iters <= args.skip_batch_num else ''
        print("Iteration: %d%s, accuracy: %f, latency: %.5f s, fps: %f" %
              (iters, appx, np.mean(infer_acc), batch_time, fps))

    # Postprocess benchmark data
    latencies = batch_times[args.skip_batch_num:]
    latency_avg = np.average(latencies)
    latency_std = np.std(latencies)
    latency_pc99 = np.percentile(latencies, 99)
    fpses = fpses[args.skip_batch_num:]
    fps_avg = np.average(fpses)
    fps_std = np.std(fpses)
    fps_pc01 = np.percentile(fpses, 1)
    infer_total_time = time.time() - infer_start_time
    examples_per_sec = total_samples / infer_total_time
    infer_accs = infer_accs[args.skip_batch_num:]
    acc_avg = np.mean(infer_accs)

    # Benchmark output
    print('\nAvg fps: %.5f, std fps: %.5f, fps for 99pc latency: %.5f' %
          (fps_avg, fps_std, fps_pc01))
    print('Avg latency: %.5f, std latency: %.5f, 99pc latency: %.5f' %
          (latency_avg, latency_std, latency_pc99))
    print('Total examples: %d, total time: %.5f, total examples/sec: %.5f' %
          (total_samples, infer_total_time, examples_per_sec))
    print("Avg accuracy: %f\n" % (acc_avg))


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    if args.data_format == 'NHWC':
        raise ValueError('Only support NCHW data_format now.')
    if args.profile:
        if args.device == 'GPU':
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                infer(args)
        else:
            with profiler.profiler(args.device, sorted_key='total') as cpuprof:
                infer(args)
    else:
        infer(args)
