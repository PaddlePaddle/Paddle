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

import cPickle
import numpy as np
import os
import sys

import paddle.fluid as fluid

PARALLEL = os.getenv('PARALLEL', '1').lower() in ['true', 't', 'on', '1']
DEVICE = os.getenv('DEVICE', 'CUDA')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
DROP_PICKLE = os.getenv('DROP_PICKLE', '1').lower() in ['true', 't', 'on', '1']
ITERATION_COUNT = int(os.getenv('ITERATION_COUNT', '20'))
FETCH_INTERVAL = int(os.getenv('FETCH_INTERVAL', '10'))

try:  # load data from file
    if DROP_PICKLE:
        raise StopIteration  # to except block
    with open('features.pkl', 'r') as f:
        random_feature, random_label, random_lod = cPickle.load(f)
except:  # random data
    batch_size = BATCH_SIZE
    lod_len = np.random.randint(low=200, high=300, size=batch_size)
    random_lod = [0]
    for l in lod_len:
        random_lod.append(random_lod[-1] + l)
    random_lod = [random_lod]
    length = random_lod[0][-1]
    random_feature = np.random.uniform(
        low=-1.0, high=1.0, size=(length, 1320)).astype('float32')
    random_label = np.random.uniform(
        low=0, high=1748, size=(length, 1)).astype('int64')
    with open('features.pkl', 'w') as f:
        cPickle.dump((random_feature, random_label, random_lod), f)


def stacked_lstm_model(hidden_dim,
                       stacked_num,
                       class_num,
                       parallel=False,
                       is_train=True):
    # network configuration
    def _net_conf(feature, label):
        seq_conv1 = fluid.layers.sequence_conv(
            input=feature,
            num_filters=1024,
            filter_size=3,
            filter_stride=1,
            bias_attr=True)
        bn1 = fluid.layers.batch_norm(
            input=seq_conv1,
            act="sigmoid",
            is_test=not is_train,
            momentum=0.9,
            epsilon=1e-05,
            data_layout='NCHW')

        stack_input = bn1
        for i in range(stacked_num):
            fc = fluid.layers.fc(input=stack_input,
                                 size=hidden_dim * 4,
                                 bias_attr=True)
            proj, cell = fluid.layers.dynamic_lstm(
                input=fc,
                size=hidden_dim * 4,
                bias_attr=True,
                use_peepholes=True,
                is_reverse=False,
                cell_activation="tanh")
            bn = fluid.layers.batch_norm(
                input=proj,
                act="sigmoid",
                is_test=not is_train,
                momentum=0.9,
                epsilon=1e-05,
                data_layout='NCHW')
            stack_input = bn

        prediction = fluid.layers.fc(input=stack_input,
                                     size=class_num,
                                     act='softmax')

        cost = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        acc = fluid.layers.accuracy(input=prediction, label=label)
        return prediction, avg_cost, acc

    # data feeder
    feature = fluid.layers.data(
        name="feature", shape=[-1, 120 * 11], dtype="float32", lod_level=1)
    label = fluid.layers.data(
        name="label", shape=[-1, 1], dtype="int64", lod_level=1)

    if parallel:
        # When the execution place is specified to CUDAPlace, the program will
        # run on all $CUDA_VISIBLE_DEVICES GPUs. Otherwise the program will
        # run on all CPU devices.
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places, use_nccl=True)
        with pd.do():
            feat_ = pd.read_input(feature)
            label_ = pd.read_input(label)
            prediction, avg_cost, acc = _net_conf(feat_, label_)
            for out in [avg_cost, acc]:
                pd.write_output(out)

        # get mean loss and acc through every devices.
        avg_cost, acc = pd()
        avg_cost = fluid.layers.mean(x=avg_cost)
        acc = fluid.layers.mean(x=acc)
    else:
        prediction, avg_cost, acc = _net_conf(feature, label)

    return prediction, avg_cost, acc


def train(stacked_num=5, hidden_dim=1024, learning_rate=0.002):
    """train in loop.
    """
    prediction, avg_cost, accuracy = stacked_lstm_model(
        hidden_dim=hidden_dim,
        stacked_num=stacked_num,
        class_num=1749,
        parallel=PARALLEL)

    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate, momentum=0.9)
    optimizer.minimize(avg_cost)

    place = fluid.CPUPlace() if DEVICE == 'CPU' else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    feature_t = fluid.LoDTensor()
    feature_t.set(random_feature, place)
    feature_t.set_lod(random_lod)
    label_t = fluid.LoDTensor()
    label_t.set(random_label, place)
    label_t.set_lod(random_lod)

    # train
    for batch_id in xrange(ITERATION_COUNT):
        # load_data
        to_print = (batch_id + 1) % FETCH_INTERVAL == 0

        result = exe.run(fluid.default_main_program(),
                         feed={"feature": feature_t,
                               "label": label_t},
                         fetch_list=[avg_cost, accuracy] if to_print else [],
                         return_numpy=False)

        if to_print:
            cost, acc = map(np.array, result)
            print("\nBatch %d, train cost: %f, train acc: %f" %
                  (batch_id, cost, acc))
        else:
            sys.stdout.write('.')
            sys.stdout.flush()


if __name__ == '__main__':
    train()
