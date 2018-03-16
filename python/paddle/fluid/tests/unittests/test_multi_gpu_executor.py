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


def vgg16_bn_drop(input):
    def conv_block(input, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=input,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=False,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    fc1 = fluid.layers.fc(input=conv5, size=4096, act=None)
    fc2 = fluid.layers.fc(input=fc1, size=4096, act=None)
    return fc2


class ParallelExecutor(unittest.TestCase):
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
        main = fluid.Program()
        startup = fluid.Program()

        with fluid.program_guard(main, startup):
            # reader = fluid.layers.open_recordio_file(
            #     filename='./mnist.recordio',
            #     shapes=[[-1, 784], [-1, 1]],
            #     lod_levels=[0, 0],
            #     dtypes=['float32', 'int64'])
            # img, label = fluid.layers.read_file(reader)
            img = fluid.layers.fill_constant(
                shape=[32, 784], dtype='float32', value=1.0)
            label = fluid.layers.fill_constant(
                shape=[32, 1], dtype='int64', value=1)
            hidden = fluid.layers.fc(img, size=2000, act='tanh')
            hidden = fluid.layers.fc(hidden, size=2000, act='tanh')
            hidden = fluid.layers.fc(hidden, size=2000, act='tanh')
            prediction = fluid.layers.fc(hidden, size=10, act='softmax')
            loss = fluid.layers.mean(prediction)
            # loss = fluid.layers.cross_entropy(input=prediction, label=label)
            # loss = fluid.layers.mean(loss)
            adam = fluid.optimizer.Adam()
            adam.minimize(loss)
        act_places = []
        for each in [fluid.CUDAPlace(0), fluid.CUDAPlace(1)]:
            p = fluid.core.Place()
            p.set_place(each)
            act_places.append(p)

        exe = fluid.core.ParallelExecutor(
            act_places,
            set([p.name for p in main.global_block().iter_parameters()]))

        exe.run(startup.desc, 0, True, True)
        exe.run(main.desc, 0, True, True)


if __name__ == '__main__':
    unittest.main()
