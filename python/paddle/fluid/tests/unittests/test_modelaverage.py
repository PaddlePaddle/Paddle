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

import unittest
import numpy as np
from op_test import OpTest
from paddle.fluid import core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
import paddle
import paddle.nn as nn


class TestModelAverage(unittest.TestCase):

    def test_model_average_static(self):
        paddle.enable_static()
        place = fluid.CPUPlace()
        shape = [2, 3, 8, 8]
        exe = fluid.Executor(place)
        train_program = fluid.Program()
        startup = fluid.Program()
        test_program = fluid.Program()
        with fluid.program_guard(train_program, startup):
            with fluid.unique_name.guard():
                data = fluid.data(name='X', shape=[None, 1], dtype='float32')
                hidden = fluid.layers.fc(input=data, size=10)
                loss = paddle.mean(hidden)
                test_program = train_program.clone()
                optimizer = paddle.optimizer.Momentum(learning_rate=0.2,
                                                      momentum=0.1)

                optimizer.minimize(loss)
                # build ModelAverage optimizer
                model_average = paddle.incubate.optimizer.ModelAverage(
                    0.15, min_average_window=2, max_average_window=10)

        exe.run(startup)
        for i in range(10):
            x = np.random.random(size=(10, 1)).astype('float32')
            latest_b, sum_1, sum_2, sum_3, num_accumulates, old_num_accumulates, num_updates = exe.run(
                program=train_program,
                feed={'X': x},
                fetch_list=[
                    'fc_0.b_0', 'fc_0.b_0_sum_1_0', 'fc_0.b_0_sum_2_0',
                    'fc_0.b_0_sum_3_0', 'fc_0.b_0_num_accumulates_0',
                    'fc_0.b_0_old_num_accumulates_0', 'fc_0.b_0_num_updates_0'
                ])
        self.assertTrue(
            np.equal(sum_1, np.zeros(shape=[10], dtype='float32')).all())
        self.assertTrue(
            np.equal(sum_2, np.zeros(shape=[10], dtype='float32')).all())
        self.assertTrue(
            np.equal(num_accumulates, np.array([0], dtype='int64')).all())
        self.assertTrue(
            np.equal(old_num_accumulates, np.array([2], dtype='int64')).all())
        self.assertTrue(
            np.equal(num_updates, np.array([10], dtype='int64')).all())

        average_b = (sum_1 + sum_2 + sum_3) / (num_accumulates +
                                               old_num_accumulates)
        # apply ModelAverage
        with model_average.apply(exe):
            x = np.random.random(size=(10, 1)).astype('float32')
            outs, b = exe.run(program=test_program,
                              feed={'X': x},
                              fetch_list=[loss.name, 'fc_0.b_0'])
            self.assertAlmostEqual(np.mean(average_b), np.mean(b))

        x = np.random.random(size=(10, 1)).astype('float32')
        outs, b = exe.run(program=test_program,
                          feed={'X': x},
                          fetch_list=[loss.name, 'fc_0.b_0'])
        self.assertAlmostEqual(np.mean(latest_b), np.mean(b))

    def test_model_average_dygraph(self):
        BATCH_SIZE = 16
        BATCH_NUM = 4
        EPOCH_NUM = 4

        IMAGE_SIZE = 784
        CLASS_NUM = 10

        # define a random dataset
        class RandomDataset(paddle.io.Dataset):

            def __init__(self, num_samples):
                self.num_samples = num_samples

            def __getitem__(self, idx):
                image = np.random.random([IMAGE_SIZE]).astype('float32')
                label = np.random.randint(0, CLASS_NUM - 1,
                                          (1, )).astype('int64')
                return image, label

            def __len__(self):
                return self.num_samples

        class LinearNet(nn.Layer):

            def __init__(self):
                super(LinearNet, self).__init__()
                self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
                self.bias = self._linear.bias

            @paddle.jit.to_static
            def forward(self, x):
                return self._linear(x)

        def train(layer, loader, loss_fn, opt, model_average):
            for epoch_id in range(EPOCH_NUM):
                for batch_id, (image, label) in enumerate(loader()):
                    out = layer(image)
                    loss = loss_fn(out, label)
                    loss.backward()
                    opt.step()
                    model_average.step()
                    opt.clear_grad()
                    model_average.clear_grad()
                    # print("Train Epoch {} batch {}: loss = {}, bias = {}".format(
                    #     epoch_id, batch_id, np.mean(loss.numpy()), layer.bias.numpy()))
            sum_1 = model_average._get_accumulator('sum_1', layer.bias)
            sum_2 = model_average._get_accumulator('sum_2', layer.bias)
            sum_3 = model_average._get_accumulator('sum_3', layer.bias)
            num_accumulates = model_average._get_accumulator(
                'num_accumulates', layer.bias)
            old_num_accumulates = model_average._get_accumulator(
                'old_num_accumulates', layer.bias)
            num_updates = model_average._get_accumulator(
                'num_updates', layer.bias)

            return ((sum_1 + sum_2 + sum_3) /
                    (num_accumulates + old_num_accumulates)).numpy()

        def evaluate(layer, loader, loss_fn, check_param):
            for batch_id, (image, label) in enumerate(loader()):
                out = layer(image)
                loss = loss_fn(out, label)
                loss.backward()
                self.assertAlmostEqual(np.mean(layer.bias.numpy()),
                                       np.mean(check_param),
                                       delta=5e-3)
                # print("Evaluate batch {}: loss = {}, bias = {}".format(
                #     batch_id, np.mean(loss.numpy()), layer.bias.numpy()))

            # create network

        layer = LinearNet()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = paddle.optimizer.Momentum(learning_rate=0.2,
                                              momentum=0.1,
                                              parameters=layer.parameters())
        # build ModelAverage optimizer
        model_average = paddle.incubate.optimizer.ModelAverage(
            0.15,
            parameters=layer.parameters(),
            min_average_window=2,
            max_average_window=10)

        # create data loader
        dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
        loader = paddle.io.DataLoader(dataset,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=2)
        eval_loader = paddle.io.DataLoader(dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=1)
        # train
        check_param = train(layer, loader, loss_fn, optimizer, model_average)
        # print(check_param)
        with model_average.apply(need_restore=False):
            evaluate(layer, eval_loader, loss_fn, check_param)

        check_param = (model_average._get_accumulator('restore',
                                                      layer.bias)).numpy()
        # print(check_param)
        # print("\nEvaluate With Restored Paramters")
        model_average.restore()
        evaluate(layer, eval_loader, loss_fn, check_param)


if __name__ == "__main__":
    unittest.main()
