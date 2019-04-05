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
import paddle.fluid.core as core
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC
from test_imperative_base import new_program_scope
from paddle.fluid.dygraph.base import to_variable


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(Discriminator, self).__init__(name_scope)
        self._fc1 = FC(self.full_name(), size=32, act='elu')
        self._fc2 = FC(self.full_name(), size=1)

    def forward(self, inputs):
        x = self._fc1(inputs)
        return self._fc2(x)


class Generator(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(Generator, self).__init__(name_scope)
        self._fc1 = FC(self.full_name(), size=64, act='elu')
        self._fc2 = FC(self.full_name(), size=64, act='elu')
        self._fc3 = FC(self.full_name(), size=1)

    def forward(self, inputs):
        x = self._fc1(inputs)
        x = self._fc2(x)
        return self._fc3(x)


class TestDygraphGAN(unittest.TestCase):
    def test_gan_float32(self):
        seed = 90

        startup = fluid.Program()
        startup.random_seed = seed
        discriminate_p = fluid.Program()
        generate_p = fluid.Program()
        discriminate_p.random_seed = seed
        generate_p.random_seed = seed

        scope = fluid.core.Scope()
        with new_program_scope(
                main=discriminate_p, startup=startup, scope=scope):
            discriminator = Discriminator("d")
            generator = Generator("g")

            img = fluid.layers.data(
                name="img", shape=[2, 1], append_batch_size=False)
            noise = fluid.layers.data(
                name="noise", shape=[2, 2], append_batch_size=False)

            d_real = discriminator(img)
            d_loss_real = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_real,
                    label=fluid.layers.fill_constant(
                        shape=[2, 1], dtype='float32', value=1.0)))

            d_fake = discriminator(generator(noise))
            d_loss_fake = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_fake,
                    label=fluid.layers.fill_constant(
                        shape=[2, 1], dtype='float32', value=0.0)))

            d_loss = d_loss_real + d_loss_fake

            sgd = SGDOptimizer(learning_rate=1e-3)
            sgd.minimize(d_loss)

        with new_program_scope(main=generate_p, startup=startup, scope=scope):
            discriminator = Discriminator("d")
            generator = Generator("g")

            noise = fluid.layers.data(
                name="noise", shape=[2, 2], append_batch_size=False)

            d_fake = discriminator(generator(noise))
            g_loss = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_fake,
                    label=fluid.layers.fill_constant(
                        shape=[2, 1], dtype='float32', value=1.0)))

            sgd = SGDOptimizer(learning_rate=1e-3)
            sgd.minimize(g_loss)

        exe = fluid.Executor(fluid.CPUPlace() if not core.is_compiled_with_cuda(
        ) else fluid.CUDAPlace(0))
        static_params = dict()
        with fluid.scope_guard(scope):
            img = np.ones([2, 1], np.float32)
            noise = np.ones([2, 2], np.float32)
            exe.run(startup)
            static_d_loss = exe.run(discriminate_p,
                                    feed={'img': img,
                                          'noise': noise},
                                    fetch_list=[d_loss])[0]
            static_g_loss = exe.run(generate_p,
                                    feed={'noise': noise},
                                    fetch_list=[g_loss])[0]

            # generate_p contains all parameters needed.
            for param in generate_p.global_block().all_parameters():
                static_params[param.name] = np.array(
                    scope.find_var(param.name).get_tensor())

        dy_params = dict()
        with fluid.dygraph.guard():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

            discriminator = Discriminator("d")
            generator = Generator("g")
            sgd = SGDOptimizer(learning_rate=1e-3)

            d_real = discriminator(to_variable(np.ones([2, 1], np.float32)))
            d_loss_real = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_real, label=to_variable(np.ones([2, 1], np.float32))))

            d_fake = discriminator(
                generator(to_variable(np.ones([2, 2], np.float32))))
            d_loss_fake = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_fake, label=to_variable(np.zeros([2, 1], np.float32))))

            d_loss = d_loss_real + d_loss_fake
            d_loss._backward()
            sgd.minimize(d_loss)
            discriminator.clear_gradients()
            generator.clear_gradients()

            d_fake = discriminator(
                generator(to_variable(np.ones([2, 2], np.float32))))
            g_loss = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_fake, label=to_variable(np.ones([2, 1], np.float32))))
            g_loss._backward()
            sgd.minimize(g_loss)
            for p in discriminator.parameters():
                dy_params[p.name] = p._numpy()
            for p in generator.parameters():
                dy_params[p.name] = p._numpy()

            dy_g_loss = g_loss._numpy()
            dy_d_loss = d_loss._numpy()

        self.assertEqual(dy_g_loss, static_g_loss)
        self.assertEqual(dy_d_loss, static_d_loss)
        for k, v in six.iteritems(dy_params):
            self.assertTrue(np.allclose(v, static_params[k]))


if __name__ == '__main__':
    unittest.main()
