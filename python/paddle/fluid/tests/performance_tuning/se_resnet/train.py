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

import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.v2.dataset.flowers as flowers
import paddle.fluid.profiler as profiler


def parse_args():
    parser = argparse.ArgumentParser('resnet152 parallel profile.')
    parser.add_argument('--per_gpu_batch_size', type=int, default=12, help='')
    parser.add_argument(
        '--skip_first_steps',
        type=int,
        default=2,
        help='The first num of steps to skip, for better performance profile')
    parser.add_argument(
        '--total_batch_num',
        type=int,
        default=40,
        help='total batch num for per_gpu_batch_size')
    parser.add_argument(
        '--parallel',
        type=distutils.util.strtobool,
        default=True,
        help='use parallel_do')
    parser.add_argument(
        '--use_nccl',
        type=distutils.util.strtobool,
        default=True,
        help='use_nccl')
    parser.add_argument(
        '--use_python_reader',
        type=distutils.util.strtobool,
        default=True,
        help='use python reader to feed data')

    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s=%s' % (arg, value))


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
    if ch_in != ch_out:
        if stride == 1:
            filter_size = 1
        else:
            filter_size = 3
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


def SE_ResNeXt(input, class_dim, infer=False):
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
        input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

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


def time_stamp():
    return int(round(time.time() * 1000))


def train():
    args = parse_args()

    cards = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    cards_num = len(cards.split(","))
    step_num = args.total_batch_num / cards_num
    batch_size = args.per_gpu_batch_size * cards_num

    print_arguments(args)
    print("cards_num=" + str(cards_num))
    print("batch_size=" + str(batch_size))
    print("total_batch_num=" + str(args.total_batch_num))
    print("step_num=" + str(step_num))

    class_dim = 1000
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if args.parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places, use_nccl=args.use_nccl)

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
        accuracy = fluid.layers.mean(x=accuracy)
    else:
        out = SE_ResNeXt(input=image, class_dim=class_dim)
        cost = fluid.layers.cross_entropy(input=out, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        accuracy = fluid.layers.accuracy(input=out, label=label)

    optimizer = fluid.optimizer.SGD(learning_rate=0.002)
    opts = optimizer.minimize(avg_cost)

    fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    train_reader = paddle.batch(flowers.train(), batch_size=batch_size)
    test_reader = paddle.batch(flowers.test(), batch_size=batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
    train_reader_iter = train_reader()
    data = train_reader_iter.next()
    feed_dict = feeder.feed(data)

    for pass_id in range(1):
        with profiler.profiler('All', 'total', '/tmp/profile') as prof:
            train_time = 0.0

            for step_id in range(step_num):
                train_start = time.time()
                if args.use_python_reader:
                    exe.run(fluid.default_main_program(),
                            feed=feeder.feed(train_reader_iter.next()),
                            fetch_list=[],
                            use_program_cache=True)
                else:
                    exe.run(fluid.default_main_program(),
                            feed=feed_dict,
                            fetch_list=[],
                            use_program_cache=True)
                train_stop = time.time()
                if step_id > args.skip_first_steps:
                    train_time += train_stop - train_start

            print("\n\n\n")
            print("train_time=" + str(train_time))


if __name__ == '__main__':
    train()
