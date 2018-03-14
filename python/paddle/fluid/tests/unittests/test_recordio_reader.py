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

import unittest
import paddle.fluid as fluid
import paddle.v2.dataset.mnist as mnist
import paddle.v2 as paddle


class TestRecordIO(unittest.TestCase):
    def setUp(self):
        # Convert mnist to recordio file
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            reader = paddle.batch(mnist.train(), batch_size=32)
            feeder = fluid.DataFeeder(
                feed_list=[  # order is image and label
                    fluid.layers.data(
                        name='image', shape=[784]),
                    fluid.layers.data(
                        name='label', shape=[1], dtype='int64'),
                ],
                place=fluid.CPUPlace())
            fluid.recordio_writer.convert_reader_to_recordio_file(
                './mnist.recordio', reader, feeder)

    def test_main(self):
        # use new program
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data_file = fluid.layers.open_recordio_file(
                './mnist.recordio',
                shapes=[[-1, 784], [-1, 1]],
                lod_levels=[0, 0],
                dtypes=['float32', 'int64'])
            img, label = fluid.layers.read_file(data_file)

            hidden = fluid.layers.fc(input=img, size=100, act='tanh')
            prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
            loss = fluid.layers.cross_entropy(input=prediction, label=label)
            avg_loss = fluid.layers.mean(loss)

            fluid.optimizer.Adam(learning_rate=1e-3).minimize(avg_loss)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            avg_loss_np = []

            # train a pass
            while not data_file.eof():
                tmp, = exe.run(fetch_list=[avg_loss])
                avg_loss_np.append(tmp)
            data_file.reset()

            self.assertLess(avg_loss_np[-1], avg_loss_np[0])
