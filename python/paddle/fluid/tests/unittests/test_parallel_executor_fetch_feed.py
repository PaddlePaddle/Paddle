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

import paddle.dataset.flowers as flowers
import math
import paddle.fluid as fluid
import unittest
import numpy as np
import paddle
import os


def Lenet(data, class_dim):
    conv1 = fluid.layers.conv2d(data, 32, 5, 1, act=None)
    bn1 = fluid.layers.batch_norm(conv1, act='relu')
    pool1 = fluid.layers.pool2d(bn1, 2, 'max', 2)
    conv2 = fluid.layers.conv2d(pool1, 50, 5, 1, act=None)
    bn2 = fluid.layers.batch_norm(conv2, act='relu')
    pool2 = fluid.layers.pool2d(bn2, 2, 'max', 2)

    fc1 = fluid.layers.fc(pool2, size=500, act='relu')
    fc2 = fluid.layers.fc(fc1, size=class_dim, act='softmax')

    return fc2


class TestFetchOp(unittest.TestCase):
    def parallel_exe(self, train_inputs, seed, use_cuda):
        main = fluid.Program()
        startup = fluid.Program()
        startup.random_seed = seed
        with fluid.program_guard(main, startup):
            data = fluid.layers.data(
                name='image', shape=[3, 224, 224], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            out = Lenet(data, class_dim=102)
            loss = fluid.layers.cross_entropy(input=out, label=label)
            loss = fluid.layers.mean(loss)

            opt = fluid.optimizer.Momentum(
                learning_rate=0.1,
                momentum=0.9,
                regularization=fluid.regularizer.L2Decay(1e-4))

            opt.minimize(loss)

            # TODO(zcd): I found that onece the memory optimizer is open,
            # parallel_exe doesn't fetch some variable, such as conv2d_0.b_0@GRAD,
            # conv2d_1.b_0@GRAD. Those variables should not be pruned.
            # fluid.memory_optimize(main)

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup)

            feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
            pe = fluid.ParallelExecutor(
                use_cuda=use_cuda, loss_name=loss.name, main_program=main)

            fetch_list = []
            all_vars = main.global_block().vars
            for k, v in all_vars.iteritems():
                if 'tmp' not in k and k[0] is not '_' or v.persistable:
                    fetch_list.append(k)

            for data in train_inputs:
                ret = pe.run(fetch_list,
                             feed=feeder.feed(data),
                             return_numpy=True)
                for i in range(len(fetch_list)):
                    assert not math.isnan(np.sum(ret[i])) and \
                           not math.isinf(np.sum(ret[i]))

    def test_fetch_op(self):
        tst_reader = paddle.batch(flowers.test(use_xmap=False), batch_size=16)
        tst_reader_iter = tst_reader()

        iters = 3
        train_inputs = []
        for i in range(iters):
            train_inputs.append(tst_reader_iter.next())

        os.environ['CPU_NUM'] = str(4)
        self.parallel_exe(train_inputs, seed=1, use_cuda=True)
        self.parallel_exe(train_inputs, seed=1, use_cuda=False)


class TestFeedParallel(unittest.TestCase):
    def parallel_exe(self, use_cuda, seed):
        main = fluid.Program()
        startup = fluid.Program()
        startup.random_seed = seed
        with fluid.scope_guard(fluid.core.Scope()):
            with fluid.program_guard(main, startup):
                data = fluid.layers.data(
                    name='image', shape=[3, 224, 224], dtype='float32')
                label = fluid.layers.data(
                    name='label', shape=[1], dtype='int64')
                out = Lenet(data, class_dim=102)
                loss = fluid.layers.cross_entropy(input=out, label=label)
                loss = fluid.layers.mean(loss)
                opt = fluid.optimizer.Momentum(
                    learning_rate=0.1,
                    momentum=0.9,
                    regularization=fluid.regularizer.L2Decay(1e-4))

                opt.minimize(loss)

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
        reader = feeder.decorate_reader(
            paddle.batch(
                flowers.train(), batch_size=16), multi_devices=True)

        exe = fluid.Executor(place)
        exe.run(startup)

        pe = fluid.ParallelExecutor(
            use_cuda=use_cuda, loss_name=loss.name, main_program=main)

        for batch_id, data in enumerate(reader()):
            loss_np = pe.run(feed=data, fetch_list=[loss.name])[0]
            print batch_id, loss_np
            if batch_id == 2:
                break

    def test_feed_op(self):
        os.environ['CPU_NUM'] = str(4)
        self.parallel_exe(use_cuda=True, seed=1)
        self.parallel_exe(use_cuda=False, seed=1)


if __name__ == '__main__':
    unittest.main()
