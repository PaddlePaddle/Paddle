# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle.fluid as fluid
import numpy as np


class TestBase(unittest.TestCase):
    def setUp(self):
        self.optimizer = fluid.optimizer.Adam

    def embedding_case(self):
        ids = fluid.layers.data(
            name='ids', dtype='int64', shape=[-1, 1], lod_level=1)
        for _ in range(2):
            ids = ids >= 0
            ids = fluid.layers.cast(ids, dtype='float32')
            ids = ids + 0.001

        ids = fluid.layers.cast(ids, dtype='int64')
        embed = fluid.layers.embedding(input=ids, size=[1024, 64])
        conv_pool = fluid.nets.sequence_conv_pool(
            input=embed, num_filters=10, filter_size=3, act='tanh')
        loss = fluid.layers.mean(conv_pool)
        optimizer = self.optimizer(learning_rate=1e-3)
        optimizer.minimize(loss)
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        for _ in range(10):
            tensor = fluid.LoDTensor()
            data = np.random.random_integers(
                low=0, high=1023, size=[256, 1]).astype('int64')
            tensor.set(data, place)
            sequence_length = list(range(data.shape[0] + 1))
            tensor.set_lod([sequence_length, ])
            exe.run(feed={'ids': tensor})

    def cross_entropy_case(self):
        image = fluid.layers.data(
            name='image', dtype='float32', shape=[-1, 784])
        label = fluid.layers.data(name='label', dtype='int64', shape=[-1, 1])

        for _ in range(10):
            label = label >= 0
            label = fluid.layers.cast(label, dtype='float32')
            label = label + 0.001

        label = fluid.layers.cast(label, dtype='int64')

        hidden = image
        for hidden_size in [20, 30, 40]:
            hidden = fluid.layers.fc(input=hidden, size=hidden_size, act='tanh')

        hidden = fluid.layers.fc(input=hidden, size=10)
        predict = fluid.layers.softmax_with_cross_entropy(hidden, label)
        loss = fluid.layers.mean(predict)

        optimizer = self.optimizer(learning_rate=1e-3)
        optimizer.minimize(loss)

        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        for _ in range(10):
            feed_image = np.random.random(size=[32, 784]).astype('float32')
            feed_label = np.random.random_integers(
                low=0, high=9, size=[32, 1]).astype('int64')
            exe.run(feed={'image': feed_image, 'label': feed_label})

    def test_cross_entropy(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            with fluid.scope_guard(fluid.Scope()):
                self.cross_entropy_case()

    def test_embedding(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            with fluid.scope_guard(fluid.Scope()):
                self.embedding_case()


class TestBase1(TestBase):
    def setUp(self):
        self.optimizer = fluid.optimizer.SGD


if __name__ == '__main__':
    unittest.main()
