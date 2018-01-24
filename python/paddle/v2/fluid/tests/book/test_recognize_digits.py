#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.v2.fluid as fluid
import paddle.v2 as paddle
import sys


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "nn_type",
        help="The neural network type, in ['mlp', 'conv']",
        type=str,
        choices=['mlp', 'conv'])
    parser.add_argument(
        "--parallel",
        help='Run in parallel or not',
        default=False,
        action="store_true")
    parser.add_argument(
        "--use_cuda",
        help="Run the program by using CUDA",
        default=False,
        action="store_true")
    return parser.parse_args()


BATCH_SIZE = 64


def loss_net(hidden, label):
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    return fluid.layers.mean(x=loss), fluid.layers.accuracy(
        input=prediction, label=label)


def mlp(img, label):
    hidden = fluid.layers.fc(input=img, size=200, act='tanh')
    hidden = fluid.layers.fc(input=hidden, size=200, act='tanh')
    return loss_net(hidden, label)


def conv_net(img, label):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    return loss_net(conv_pool_2, label)


def main():
    args = parse_arg()
    print("recognize digits with args: {0}".format(" ".join(sys.argv[1:])))

    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if args.nn_type == 'mlp':
        net_conf = mlp
    else:
        net_conf = conv_net

    if args.parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            img_ = pd.read_input(img)
            label_ = pd.read_input(label)
            for o in net_conf(img_, label_):
                pd.write_output(o)

        avg_loss, acc = pd()
        # get mean loss and acc through every devices.
        avg_loss = fluid.layers.mean(x=avg_loss)
        acc = fluid.layers.mean(x=acc)
    else:
        avg_loss, acc = net_conf(img, label)

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    PASS_NUM = 100
    for pass_id in range(PASS_NUM):
        for batch_id, data in enumerate(train_reader()):
            need_check = (batch_id + 1) % 10 == 0

            if need_check:
                fetch_list = [avg_loss, acc]
            else:
                fetch_list = []

            outs = exe.run(feed=feeder.feed(data), fetch_list=fetch_list)
            if need_check:
                avg_loss_np, acc_np = outs
                if float(acc_np) > 0.9:
                    exit(0)
                else:
                    print(
                        'PassID {0:1}, BatchID {1:04}, Loss {2:2.2}, Acc {3:2.2}'.
                        format(pass_id, batch_id + 1,
                               float(avg_loss_np), float(acc_np)))


if __name__ == '__main__':
    main()
