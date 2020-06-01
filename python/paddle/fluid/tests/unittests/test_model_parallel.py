# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
import os
import paddle
import paddle.fluid as fluid
import numpy
import unittest

NUM_MICRO = 2
BATCH_SIZE = 8

fluid.default_startup_program().random_seed = 32
fluid.default_main_program().random_seed = 32


def reader():
    numpy.random.seed(10)
    for idx in range():
        all_data = []
        for _ in range(80):
            diff = numpy.random.random() - 0.4
            diff = max(0.0, diff)
            data = float(idx) - diff
            all_data.append(data)
        label = (10 * idx) % 10
        yield all_data, label


def get_model():
    input_dim = 10
    input = fluid.layers.data(name="data_1", shape=[10], dtype="float32")
    input_orig = input
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    with fluid.device_guard("gpu:0"):
        data_loader = fluid.io.DataLoader.from_generator(
            feed_list=[input, label],
            capacity=64,
            use_double_buffer=False,
            iterable=False)
        input = fluid.layers.fc(
            input=input,
            size=input_dim,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(0, 0.01)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer(0.0)))
    with fluid.device_guard("gpu:1"):
        out = fluid.layers.softmax_with_cross_entropy(logits=input, label=label)
        loss = fluid.layers.mean(out)
    return loss, input_orig, label


class TestModelParallel(unittest.TestCase):
    def test_modelparallel(self):
        loss, input, label = get_model()
        opt = fluid.optimizer.Momentum(0.1, momentum=0.9)
        opt = fluid.optimizer.ModelParallelOptimizer(
            opt, num_macrobatches=NUM_MICRO)
        opt.minimize(loss)

        batch_size = BATCH_SIZE
        batch_size = batch_size // NUM_MICRO
        data_reader = paddle.batch(reader, batch_size=batch_size)
        place = fluid.CUDAPlace(0)
        data_loader.set_sample_list_generator(data_reader, place)
        data_loader.start()

        dataset = fluid.DatasetFactory().create_dataset('FileInstantDataset')
        dataset.set_batch_size(1)
        dataset.set_thread(1)
        dataset.set_filelist(['/tmp/tmp.txt'])
        dataset.set_use_var([input, label])
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        exe.train_from_dataset(
            fluid.default_main_program(), dataset, debug=False)


if __name__ == '__main__':
    unittest.main()
