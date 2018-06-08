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

import os
import time
import argparse
import distutils.util
import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.dataset.flowers as flowers
import paddle.fluid.profiler as profiler
from memory_transpiler import SSAGraph

fluid.default_startup_program().random_seed = 111

def parse_args():
    parser = argparse.ArgumentParser('SE-ResNeXt-152 parallel profile.')
    parser.add_argument(
        '--parallel_mode',
        type=str,
        default='parallel_exe',
        choices=['parallel_do', 'parallel_exe'],
        help='The parallel mode("parallel_do" or "parallel_exe").')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--batch_size_per_gpu', type=int, default=16, help='')
    parser.add_argument(
        '--use_mem_opt',
        action='store_true',
        help='use memory optimize or not.')
    parser.add_argument(
        '--do_profile', action='store_true', help='do profile or not.')
    parser.add_argument('--number_iteration', type=int, default=150, help='')
    parser.add_argument('--display_step', type=int, default=10, help='')
    parser.add_argument('--skip_first_steps', type=int, default=30, help='.')
    parser.add_argument('--fix_data_in_gpu', action='store_true', help='')
    parser.add_argument('--use_recordio', action='store_true', help='.')
    parser.add_argument(
        '--balance_parameter_opt_between_cards',
        action='store_true',
        help='balance parameter opt between cards')
    parser.add_argument('--show_record_time', action='store_true', help='')
    parser.add_argument('--use_fake_reader', action='store_true', help='')

    args = parser.parse_args()
    return args



def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1,
                  act=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) / 2,
        groups=groups,
        act=None,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv, act=act)


def squeeze_excitation(input, num_channels, reduction_ratio):
    pool = fluid.layers.pool2d(
        input=input, pool_size=0, pool_type='avg', global_pooling=True)
    squeeze = fluid.layers.fc(input=pool,
                              size=num_channels / reduction_ratio,
                              act='relu')
    excitation = fluid.layers.fc(input=squeeze,
                                 size=num_channels,
                                 act='sigmoid')
    scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
    return scale


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]
    if ch_in != ch_out or stride != 1:
        filter_size = 1
        return conv_bn_layer(input, ch_out, filter_size, stride)
    else:
        return input


def bottleneck_block(input, num_filters, stride, cardinality, reduction_ratio):
    conv0 = conv_bn_layer(
        input=input, num_filters=num_filters, filter_size=1, act='relu')
    conv1 = conv_bn_layer(
        input=conv0,
        num_filters=num_filters,
        filter_size=3,
        stride=stride,
        groups=cardinality,
        act='relu')
    conv2 = conv_bn_layer(
        input=conv1, num_filters=num_filters * 2, filter_size=1, act=None)
    scale = squeeze_excitation(
        input=conv2,
        num_channels=num_filters * 2,
        reduction_ratio=reduction_ratio)

    short = shortcut(input, num_filters * 2, stride)
    return fluid.layers.elementwise_add(x=short, y=scale, act='relu')


def SE_ResNeXt(input, class_dim, infer=False, layers=152):
    supported_layers = [50, 152]
    if layers not in supported_layers:
        print("supported layers are", supported_layers, "but input layer is ",
              layers)
        exit()
    if layers == 50:
        cardinality = 32
        reduction_ratio = 16
        depth = [3, 4, 6, 3]
        num_filters = [128, 256, 512, 1024]

        conv = conv_bn_layer(
            input=input, num_filters=64, filter_size=7, stride=2, act='relu')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
    elif layers == 152:
        cardinality = 64
        reduction_ratio = 16
        depth = [3, 8, 36, 3]
        num_filters = [128, 256, 512, 1024]

        conv = conv_bn_layer(
            input=input, num_filters=64, filter_size=3, stride=2, act='relu')
        conv = conv_bn_layer(
            input=conv, num_filters=64, filter_size=3, stride=1, act='relu')
        conv = conv_bn_layer(
            input=conv, num_filters=128, filter_size=3, stride=1, act='relu')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv = bottleneck_block(
                input=conv,
                num_filters=num_filters[block],
                stride=2 if i == 0 and block != 0 else 1,
                cardinality=cardinality,
                reduction_ratio=reduction_ratio)

    pool = fluid.layers.pool2d(
        input=conv, pool_size=0, pool_type='avg', global_pooling=True)
    if not infer:
        drop = fluid.layers.dropout(x=pool, dropout_prob=0.2)
    else:
        drop = pool
    out = fluid.layers.fc(input=drop, size=class_dim, act='softmax')
    return out


def net_conf(image, label, class_dim):
    out = SE_ResNeXt(input=image, class_dim=class_dim)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    accuracy = fluid.layers.accuracy(input=out, label=label)
    accuracy5 = fluid.layers.accuracy(input=out, label=label, k=5)
    return out, avg_cost, accuracy, accuracy5


def add_optimizer(args, avg_cost):
    #optimizer = fluid.optimizer.SGD(learning_rate=0.002)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=[100], values=[0.1, 0.2]),
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(1e-4))
    optimizer.minimize(avg_cost)

    if args.use_mem_opt:
        fluid.memory_optimize(fluid.default_main_program())


def fake_reader():
    while True:
        img = np.random.rand(3, 224, 224)
        lab = np.random.randint(0, 999)
        yield img, lab


def train():
    return fake_reader


def train_parallel_do(args):

    class_dim = 1000
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    places = fluid.layers.get_places()
    pd = fluid.layers.ParallelDo(places, use_nccl=False)

    with pd.do():
        image_ = pd.read_input(image)
        label_ = pd.read_input(label)
        out = SE_ResNeXt(input=image_, class_dim=class_dim)
        cost = fluid.layers.cross_entropy(input=out, label=label_)
        avg_cost = fluid.layers.mean(x=cost)
        accuracy = fluid.layers.accuracy(input=out, label=label_)
        pd.write_output(avg_cost)
        pd.write_output(accuracy)

    avg_cost, accuracy = pd()
    avg_cost = fluid.layers.mean(x=avg_cost)
    # accuracy = fluid.layers.mean(x=accuracy)

    add_optimizer(args, avg_cost)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    ops = fluid.default_startup_program().block(0).ops + fluid.default_main_program().block(0).ops
    ssa = SSAGraph()
    print(len(ops))
    fluid.memory_optimize(fluid.default_main_program(), print_log=True)
    # ssa._run_graph(ops)
    exit(0)

    train_reader = paddle.batch(
        train() if args.use_fake_reader else flowers.train(),
        batch_size=args.batch_size)

    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
    train_reader_iter = train_reader()
    if args.fix_data_in_gpu:
        data = train_reader_iter.next()
        feed_data = feeder.feed(data)

    time_record = []
    img_count = 0
    train_start = time.time()

    for batch_id in range(args.number_iteration):
        if args.do_profile and batch_id >= 5 and batch_id < 8:
            with profiler.profiler('All', 'total',
                                   '/tmp/profile_parallel_do') as prof:
                exe.run(fluid.default_main_program(),
                        feed=feed_data if args.fix_data_in_gpu else
                        feeder.feed(train_reader_iter.next()),
                        fetch_list=[],
                        use_program_cache=True)
            continue

        cost_val = exe.run(fluid.default_main_program(),
                           feed=feed_data if args.fix_data_in_gpu else
                           feeder.feed(train_reader_iter.next()),
                           fetch_list=[avg_cost.name]
                           if (batch_id + 1) % args.display_step == 0 else [],
                           use_program_cache=True)

        img_count += args.batch_size

        if (batch_id + 1) % args.display_step == 0:
            train_stop = time.time()
            step_time = train_stop - train_start
            time_record.append(step_time)

            print("iter=%d, cost=%s, elapse=%f, img/sec=%f" %
                  ((batch_id + 1), np.array(cost_val[0]), step_time,
                   img_count / step_time))

            img_count = 0
            train_start = time.time()

    skip_time_record = args.skip_first_steps / args.display_step
    time_record[0:skip_time_record] = []

    if args.show_record_time:
        for i, ele in enumerate(time_record):
            print("iter:{0}, time consume:{1}".format(i, ele))

    img_count = (
        args.number_iteration - args.skip_first_steps) * args.batch_size

    print("average time:{0}, img/sec:{1}".format(
        np.mean(time_record), img_count / np.sum(time_record)))



if __name__ == '__main__':

    args = parse_args()
    cards = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    cards_num = len(cards.split(","))
    train_parallel_do(args)
