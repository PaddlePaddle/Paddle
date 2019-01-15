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

import contextlib
import unittest
import numpy as np
import six
import sys

import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.imperative.nn import Conv2D, Pool2D, FC
from test_imperative_base import new_program_scope


class Discriminator(fluid.imperative.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()
        self._fc1 = FC(size=32, act='elu', name="d_fc1")
        self._fc2 = FC(size=1, name="d_fc2")

    def forward(self, inputs):
        x = self._fc1(inputs)
        return self._fc2(x)


class Generator(fluid.imperative.Layer):
    def __init__(self):
        super(Generator, self).__init__()
        self._fc1 = FC(size=64, act='elu', name="g_fc1")
        self._fc2 = FC(size=64, act='elu', name="g_fc2")
        self._fc3 = FC(size=1, name="g_fc3")

    def forward(self, inputs):
        x = self._fc1(inputs)
        x = self._fc2(x)
        return self._fc3(x)


class TestImperativeMnist(unittest.TestCase):
    def test_mnist_cpu_float32(self):
        seed = 90

        startup = fluid.Program()
        startup.random_seed = seed
        discriminate_p = fluid.Program()
        scope = fluid.core.Scope()
        exe = fluid.Executor(fluid.CPUPlace())
        with new_program_scope(
                main=discriminate_p, startup=startup, scope=scope):
            fluid.default_main_program().random_seed = seed

            discriminator = Discriminator()
            generator = Generator()

            img = fluid.layers.data(
                name="img", shape=[2, 1], append_batch_size=False)
            noise = fluid.layers.data(
                name="noise", shape=[2, 2], append_batch_size=False)

            label = fluid.layers.data(
                name='label',
                shape=[2, 1],
                dtype='float32',
                append_batch_size=False)

            d_real = discriminator(img)
            d_loss_real = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_real, label=label))

            d_fake = discriminator(generator(noise))
            d_loss_fake = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_fake, label=label))

            d_loss = d_loss_real + d_loss_fake

            sgd = SGDOptimizer(learning_rate=1e-3)
            sgd.minimize(d_loss)

        generate_p = fluid.Program()
        with new_program_scope(main=generate_p, startup=startup, scope=scope):
            fluid.default_main_program().random_seed = seed

            discriminator = Discriminator()
            generator = Generator()

            noise = fluid.layers.data(
                name="noise", shape=[2, 2], append_batch_size=False)
            label = fluid.layers.data(
                name='label',
                shape=[2, 1],
                dtype='float32',
                append_batch_size=False)

            d_fake = discriminator(generator(noise))
            g_loss = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_fake, label=label))

            sgd = SGDOptimizer(learning_rate=1e-3)
            sgd.minimize(g_loss)

        img = np.ones([2, 1], np.float32)
        label = np.ones([2, 1], np.float32)
        noise = np.ones([2, 2], np.float32)
        exe.run(startup)
        d_loss_val = exe.run(discriminate_p,
                             feed={'img': img,
                                   'noise': noise,
                                   'label': label},
                             fetch_list=[d_loss])[0]
        g_loss_val = exe.run(generate_p,
                             feed={'noise': noise,
                                   'label': label},
                             fetch_list=[g_loss])[0]
        sys.stderr.write('d_loss %s, g_loss: %s\n' % (d_loss_val, g_loss_val))


if __name__ == '__main__':
    unittest.main()
