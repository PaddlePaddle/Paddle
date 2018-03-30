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
            self.num_batches = fluid.recordio_writer.convert_reader_to_recordio_file(
                './mnist.recordio', reader, feeder)

    def test_main(self, decorator_callback=None):
        # use new program
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data_file = fluid.layers.open_recordio_file(
                './mnist.recordio',
                shapes=[[-1, 784], [-1, 1]],
                lod_levels=[0, 0],
                dtypes=['float32', 'int64'])
            if decorator_callback is not None:
                data_file = decorator_callback(data_file)
            img, label = fluid.layers.read_file(data_file)

            hidden = fluid.layers.fc(input=img, size=100, act='tanh')
            prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
            loss = fluid.layers.cross_entropy(input=prediction, label=label)
            avg_loss = fluid.layers.mean(loss)

            fluid.optimizer.Adam(learning_rate=1e-3).minimize(avg_loss)

            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            else:
                place = fluid.CPUPlace()

            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            avg_loss_np = []

            # train a pass
            batch_id = 0
            while not data_file.eof():
                tmp, = exe.run(fetch_list=[avg_loss])
                avg_loss_np.append(tmp)
                batch_id += 1
            data_file.reset()
            self.assertEqual(batch_id, self.num_batches)
            self.assertLess(avg_loss_np[-1], avg_loss_np[0])

    def test_shuffle_reader(self):
        self.test_main(decorator_callback=lambda reader: fluid.layers.create_shuffle_reader(reader, buffer_size=200))

    def test_double_buffer_reader(self):
        self.test_main(decorator_callback=lambda reader: fluid.layers.create_double_buffer_reader(reader,
                                                                                                  place='cuda:0' if fluid.core.is_compiled_with_cuda() else 'cpu'))


import paddle.v2.dataset.flowers as flowers


def vgg16_bn_drop(input):
    def conv_block(input, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=input,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    return fc2


class TestDoubleBuffer(unittest.TestCase):
    BATCH_SIZE = 4

    @classmethod
    def setUpClass(cls):
        # import os
        # if os.path.exists('./flowers.recordio'):
        #     return
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            reader = paddle.batch(
                flowers.train(), batch_size=TestDoubleBuffer.BATCH_SIZE)
            feeder = fluid.DataFeeder(
                feed_list=[
                    fluid.layers.data(
                        name='image', shape=[3, 224, 224]),
                    fluid.layers.data(
                        name='label', shape=[1], dtype='int64'),
                ],
                place=fluid.CPUPlace())
            fluid.recordio_writer.convert_reader_to_recordio_file(
                "./flowers_double_buffer.recordio",
                reader,
                feeder,
                compressor=fluid.core.RecordIOWriter.Compressor.NoCompress)

    def test_main(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data_file = fluid.layers.open_recordio_file(
                filename='./flowers_double_buffer.recordio',
                shapes=[[-1, 3, 224, 224], [-1, 1]],
                lod_levels=[0, 0],
                dtypes=['float32', 'int64'])
            data_file = fluid.layers.create_double_buffer_reader(
                reader=data_file, place='CUDA:0')
            images, label = fluid.layers.read_file(data_file)

            net = vgg16_bn_drop(images)
            predict = fluid.layers.fc(input=net, size=1000, act='softmax')
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            avg_cost = fluid.layers.mean(x=cost)
            optimizer = fluid.optimizer.Adam(learning_rate=1e-4)
            optimizer.minimize(avg_cost)

            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)

            # Parameter initialization
            iters = 0
            exe.run(fluid.default_startup_program())
            while not data_file.eof():
                # for batch_id, data in enumerate(train_reader()):
                loss, = exe.run(fluid.default_main_program(),
                                fetch_list=[avg_cost])
                print "Iters = %d, Loss = %f" % (iters, loss)
                iters += 1
