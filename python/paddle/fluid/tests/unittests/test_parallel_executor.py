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
import paddle.v2 as paddle
import paddle.v2.dataset.mnist as mnist
import numpy


def simple_fc_net():
    reader = fluid.layers.open_recordio_file(
        filename='./mnist.recordio',
        shapes=[[-1, 784], [-1, 1]],
        lod_levels=[0, 0],
        dtypes=['float32', 'int64'])
    img, label = fluid.layers.read_file(reader)
    hidden = img
    for _ in xrange(4):
        hidden = fluid.layers.fc(
            hidden,
            size=200,
            act='tanh',
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=1.0)))
    prediction = fluid.layers.fc(hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    return loss


def fc_with_batchnorm():
    reader = fluid.layers.open_recordio_file(
        filename='./mnist.recordio',
        shapes=[[-1, 784], [-1, 1]],
        lod_levels=[0, 0],
        dtypes=['float32', 'int64'])
    img, label = fluid.layers.read_file(reader)
    hidden = img
    for _ in xrange(4):
        hidden = fluid.layers.fc(
            hidden,
            size=200,
            act='tanh',
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=1.0)))

        hidden = fluid.layers.batch_norm(input=hidden)

    prediction = fluid.layers.fc(hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    return loss


class ParallelExecutor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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

    def test_simple_fc(self):
        self.check_network_convergence(simple_fc_net)

    def test_batchnorm_fc(self):
        self.check_network_convergence(fc_with_batchnorm)

    def check_network_convergence(self, method):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = method()
            adam = fluid.optimizer.Adam()
            adam.minimize(loss)
            exe = fluid.ParallelExecutor(loss_name=loss.name, use_cuda=True)
            first_loss, = exe.run([loss.name])
            first_loss = numpy.array(first_loss)

            for i in xrange(10):
                exe.run([])

            last_loss, = exe.run([loss.name])
            last_loss = numpy.array(last_loss)

            print first_loss, last_loss
            self.assertGreater(first_loss[0], last_loss[0])
