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

import paddle.fluid as fluid
import numpy
import sys

TRAIN_FILES = ['train.recordio']
TEST_FILES = ['test.recordio']

DICT_DIM = 5147

# embedding dim
emb_dim = 128

# hidden dim
hid_dim = 128

# class num
class_dim = 2

# epoch num
epoch_num = 10


def build_program(is_train):
    file_obj_handle = fluid.layers.io.open_files(
        filenames=TRAIN_FILES if is_train else TEST_FILES,
        shapes=[[-1, 1], [-1, 1]],
        lod_levels=[1, 0],
        dtypes=['int64', 'int64'])

    file_obj = fluid.layers.io.double_buffer(file_obj_handle)

    with fluid.unique_name.guard():

        data, label = fluid.layers.read_file(file_obj)

        emb = fluid.layers.embedding(input=data, size=[DICT_DIM, emb_dim])

        conv_3 = fluid.nets.sequence_conv_pool(
            input=emb,
            num_filters=hid_dim,
            filter_size=3,
            act="tanh",
            pool_type="sqrt")

        conv_4 = fluid.nets.sequence_conv_pool(
            input=emb,
            num_filters=hid_dim,
            filter_size=4,
            act="tanh",
            pool_type="sqrt")

        prediction = fluid.layers.fc(input=[conv_3, conv_4],
                                     size=class_dim,
                                     act="softmax")

        # cross entropy loss
        cost = fluid.layers.cross_entropy(input=prediction, label=label)

        # mean loss
        avg_cost = fluid.layers.mean(x=cost)
        acc = fluid.layers.accuracy(input=prediction, label=label)

        if is_train:
            # SGD optimizer
            sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=0.001)
            sgd_optimizer.minimize(avg_cost)

    return {'loss': avg_cost, 'log': [avg_cost, acc], 'file': file_obj_handle}


def main():
    train = fluid.Program()
    startup = fluid.Program()
    test = fluid.Program()

    with fluid.program_guard(train, startup):
        train_args = build_program(is_train=True)

    with fluid.program_guard(test, startup):
        test_args = build_program(is_train=False)

    use_cuda = fluid.core.is_compiled_with_cuda()
    # startup
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place=place)
    exe.run(startup)

    train_exe = fluid.ParallelExecutor(
        use_cuda=use_cuda,
        loss_name=train_args['loss'].name,
        main_program=train)
    test_exe = fluid.ParallelExecutor(
        use_cuda=use_cuda, main_program=test, share_vars_from=train_exe)

    fetch_var_list = [var.name for var in train_args['log']]
    for epoch_id in range(epoch_num):
        # train
        try:
            batch_id = 0
            while True:
                loss, acc = map(numpy.array,
                                train_exe.run(fetch_list=fetch_var_list))
                print 'Train epoch', epoch_id, 'batch', batch_id, 'loss:', loss, 'acc:', acc
                batch_id += 1
        except fluid.core.EOFException:
            print 'End of epoch', epoch_id
            train_args['file'].reset()

        # test
        loss = []
        acc = []
        try:
            while True:
                loss_np, acc_np = map(numpy.array,
                                      test_exe.run(fetch_list=fetch_var_list))
                loss.append(loss_np[0])
                acc.append(acc_np[0])
        except:
            test_args['file'].reset()
            print 'Test loss:', numpy.mean(loss), 'acc:', numpy.mean(acc)


if __name__ == '__main__':
    main()
