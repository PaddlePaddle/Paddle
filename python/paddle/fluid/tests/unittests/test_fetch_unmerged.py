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
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid

os.environ["CPU_NUM"] = "2"


class TestFetchUnmerged(unittest.TestCase):
    def conv_net(self, img, label):
        conv_pool_1 = fluid.nets.simple_img_conv_pool(
            input=img,
            filter_size=5,
            num_filters=8,
            pool_size=2,
            pool_stride=2,
            pool_type='max',
            act="relu",
        )
        conv_pool_1 = paddle.static.nn.batch_norm(conv_pool_1)
        conv_pool_2 = fluid.nets.simple_img_conv_pool(
            input=conv_pool_1,
            filter_size=5,
            num_filters=16,
            pool_size=2,
            pool_stride=2,
            pool_type='avg',
            act="relu",
        )
        hidden = fluid.layers.fc(input=conv_pool_2, size=32, act='relu')
        prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
        loss = paddle.nn.functional.cross_entropy(
            input=prediction, label=label, reduction='none', use_softmax=False
        )
        avg_loss = paddle.mean(loss)
        return avg_loss, prediction

    def build_program(self, main, startup, is_test):
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
                img = fluid.layers.data(
                    name='image', shape=[1, 28, 28], dtype='float32'
                )
                label = fluid.layers.data(
                    name='label', shape=[1], dtype='int64'
                )
                loss, prediction = self.conv_net(img, label)
                if not is_test:
                    opt = fluid.optimizer.Adam(learning_rate=0.001)
                    opt.minimize(loss)
        return [img, label], loss, prediction

    def fetch_unmerged(self, use_cuda=True):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        feeds, loss, prediction = self.build_program(
            main_program, startup_program, False
        )

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)

        build_strategy = fluid.BuildStrategy()
        binary = fluid.CompiledProgram(main_program).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy
        )

        iters = 2
        batch_size = 16
        train_reader = paddle.batch(
            paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
            batch_size=batch_size,
        )
        feeder = fluid.DataFeeder(feed_list=feeds, place=place)

        device_num = fluid.core.get_cuda_device_count() if use_cuda else 2
        for _ in range(iters):
            data = next(train_reader())
            loss_v, prediction_v = exe.run(
                binary,
                feed=feeder.feed(data),
                fetch_list=[loss, prediction],
                return_merged=False,
            )
            self.assertEqual(np.array(loss_v).shape, (device_num, 1))
            self.assertEqual(
                np.array(prediction_v).shape,
                (device_num, batch_size / device_num, 10),
            )

        for _ in range(iters):
            data = next(train_reader())
            loss_v, prediction_v = exe.run(
                binary,
                feed=feeder.feed(data),
                fetch_list=[loss, prediction],
                return_merged=True,
            )
            self.assertEqual(np.array(loss_v).shape, (device_num,))
            self.assertEqual(np.array(prediction_v).shape, (batch_size, 10))

    def test_fetch_unmerged(self):
        if fluid.core.is_compiled_with_cuda():
            self.fetch_unmerged(use_cuda=True)
        self.fetch_unmerged(use_cuda=False)

    def test_fetch_unmerged_parallel_graph(self):
        fluid.core.globals()['FLAGS_enable_parallel_graph'] = True
        if fluid.core.is_compiled_with_cuda():
            self.fetch_unmerged(use_cuda=True)
        self.fetch_unmerged(use_cuda=False)
        fluid.core.globals()['FLAGS_enable_parallel_graph'] = False


if __name__ == '__main__':
    unittest.main()
