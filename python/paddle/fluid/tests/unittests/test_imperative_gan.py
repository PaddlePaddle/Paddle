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

<<<<<<< HEAD
import unittest

import numpy as np
from test_imperative_base import new_program_scope
=======
import contextlib
import unittest
import numpy as np
import six
import sys
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
<<<<<<< HEAD
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.optimizer import SGDOptimizer
from paddle.nn import Linear


class Discriminator(fluid.Layer):
    def __init__(self):
        super().__init__()
        self._fc1 = Linear(1, 32)
=======
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid import Conv2D, Pool2D, Linear
from test_imperative_base import new_program_scope
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.framework import _test_eager_guard


class Discriminator(fluid.Layer):

    def __init__(self):
        super(Discriminator, self).__init__()
        self._fc1 = Linear(1, 32, act='elu')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._fc2 = Linear(32, 1)

    def forward(self, inputs):
        x = self._fc1(inputs)
<<<<<<< HEAD
        x = paddle.nn.functional.elu(x)
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x = self._fc2(x)
        return x


class Generator(fluid.Layer):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
        self._fc1 = Linear(2, 64)
        self._fc2 = Linear(64, 64)
=======

    def __init__(self):
        super(Generator, self).__init__()
        self._fc1 = Linear(2, 64, act='elu')
        self._fc2 = Linear(64, 64, act='elu')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._fc3 = Linear(64, 1)

    def forward(self, inputs):
        x = self._fc1(inputs)
<<<<<<< HEAD
        x = paddle.nn.functional.elu(x)
        x = self._fc2(x)
        x = paddle.nn.functional.elu(x)
=======
        x = self._fc2(x)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x = self._fc3(x)
        return x


class TestDygraphGAN(unittest.TestCase):
<<<<<<< HEAD
    def test_gan_float32(self):
=======

    def func_test_gan_float32(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        seed = 90
        paddle.seed(1)
        paddle.framework.random._manual_program_seed(1)
        startup = fluid.Program()
        discriminate_p = fluid.Program()
        generate_p = fluid.Program()

        scope = fluid.core.Scope()
<<<<<<< HEAD
        with new_program_scope(
            main=discriminate_p, startup=startup, scope=scope
        ):
            discriminator = Discriminator()
            generator = Generator()

            img = paddle.static.data(name="img", shape=[2, 1])
            noise = paddle.static.data(name="noise", shape=[2, 2])

            d_real = discriminator(img)
            d_loss_real = paddle.mean(
                paddle.nn.functional.binary_cross_entropy_with_logits(
                    logit=d_real,
                    label=fluid.layers.fill_constant(
                        shape=[2, 1], dtype='float32', value=1.0
                    ),
                )
            )

            d_fake = discriminator(generator(noise))
            d_loss_fake = paddle.mean(
                paddle.nn.functional.binary_cross_entropy_with_logits(
                    logit=d_fake,
                    label=fluid.layers.fill_constant(
                        shape=[2, 1], dtype='float32', value=0.0
                    ),
                )
            )
=======
        with new_program_scope(main=discriminate_p,
                               startup=startup,
                               scope=scope):
            discriminator = Discriminator()
            generator = Generator()

            img = fluid.layers.data(name="img",
                                    shape=[2, 1],
                                    append_batch_size=False)
            noise = fluid.layers.data(name="noise",
                                      shape=[2, 2],
                                      append_batch_size=False)

            d_real = discriminator(img)
            d_loss_real = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_real,
                    label=fluid.layers.fill_constant(shape=[2, 1],
                                                     dtype='float32',
                                                     value=1.0)))

            d_fake = discriminator(generator(noise))
            d_loss_fake = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_fake,
                    label=fluid.layers.fill_constant(shape=[2, 1],
                                                     dtype='float32',
                                                     value=0.0)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            d_loss = d_loss_real + d_loss_fake

            sgd = SGDOptimizer(learning_rate=1e-3)
            sgd.minimize(d_loss)

        with new_program_scope(main=generate_p, startup=startup, scope=scope):
            discriminator = Discriminator()
            generator = Generator()

<<<<<<< HEAD
            noise = paddle.static.data(name="noise", shape=[2, 2])

            d_fake = discriminator(generator(noise))
            g_loss = paddle.mean(
                paddle.nn.functional.binary_cross_entropy_with_logits(
                    logit=d_fake,
                    label=fluid.layers.fill_constant(
                        shape=[2, 1], dtype='float32', value=1.0
                    ),
                )
            )
=======
            noise = fluid.layers.data(name="noise",
                                      shape=[2, 2],
                                      append_batch_size=False)

            d_fake = discriminator(generator(noise))
            g_loss = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_fake,
                    label=fluid.layers.fill_constant(shape=[2, 1],
                                                     dtype='float32',
                                                     value=1.0)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            sgd = SGDOptimizer(learning_rate=1e-3)
            sgd.minimize(g_loss)

<<<<<<< HEAD
        exe = fluid.Executor(
            fluid.CPUPlace()
            if not core.is_compiled_with_cuda()
            else fluid.CUDAPlace(0)
        )
=======
        exe = fluid.Executor(fluid.CPUPlace(
        ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        static_params = dict()
        with fluid.scope_guard(scope):
            img = np.ones([2, 1], np.float32)
            noise = np.ones([2, 2], np.float32)
            exe.run(startup)
<<<<<<< HEAD
            static_d_loss = exe.run(
                discriminate_p,
                feed={'img': img, 'noise': noise},
                fetch_list=[d_loss],
            )[0]
            static_g_loss = exe.run(
                generate_p, feed={'noise': noise}, fetch_list=[g_loss]
            )[0]
=======
            static_d_loss = exe.run(discriminate_p,
                                    feed={
                                        'img': img,
                                        'noise': noise
                                    },
                                    fetch_list=[d_loss])[0]
            static_g_loss = exe.run(generate_p,
                                    feed={'noise': noise},
                                    fetch_list=[g_loss])[0]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            # generate_p contains all parameters needed.
            for param in generate_p.global_block().all_parameters():
                static_params[param.name] = np.array(
<<<<<<< HEAD
                    scope.find_var(param.name).get_tensor()
                )
=======
                    scope.find_var(param.name).get_tensor())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        dy_params = dict()
        with fluid.dygraph.guard():
            paddle.seed(1)
            paddle.framework.random._manual_program_seed(1)

            discriminator = Discriminator()
            generator = Generator()
<<<<<<< HEAD
            sgd = SGDOptimizer(
                learning_rate=1e-3,
                parameter_list=(
                    discriminator.parameters() + generator.parameters()
                ),
            )

            d_real = discriminator(to_variable(np.ones([2, 1], np.float32)))
            d_loss_real = paddle.mean(
                paddle.nn.functional.binary_cross_entropy_with_logits(
                    logit=d_real, label=to_variable(np.ones([2, 1], np.float32))
                )
            )

            d_fake = discriminator(
                generator(to_variable(np.ones([2, 2], np.float32)))
            )
            d_loss_fake = paddle.mean(
                paddle.nn.functional.binary_cross_entropy_with_logits(
                    logit=d_fake,
                    label=to_variable(np.zeros([2, 1], np.float32)),
                )
            )
=======
            sgd = SGDOptimizer(learning_rate=1e-3,
                               parameter_list=(discriminator.parameters() +
                                               generator.parameters()))

            d_real = discriminator(to_variable(np.ones([2, 1], np.float32)))
            d_loss_real = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_real, label=to_variable(np.ones([2, 1], np.float32))))

            d_fake = discriminator(
                generator(to_variable(np.ones([2, 2], np.float32))))
            d_loss_fake = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_fake, label=to_variable(np.zeros([2, 1], np.float32))))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            sgd.minimize(d_loss)
            discriminator.clear_gradients()
            generator.clear_gradients()

            d_fake = discriminator(
<<<<<<< HEAD
                generator(to_variable(np.ones([2, 2], np.float32)))
            )
            g_loss = paddle.mean(
                paddle.nn.functional.binary_cross_entropy_with_logits(
                    logit=d_fake, label=to_variable(np.ones([2, 1], np.float32))
                )
            )
=======
                generator(to_variable(np.ones([2, 2], np.float32))))
            g_loss = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_fake, label=to_variable(np.ones([2, 1], np.float32))))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            g_loss.backward()
            sgd.minimize(g_loss)
            for p in discriminator.parameters():
                dy_params[p.name] = p.numpy()
            for p in generator.parameters():
                dy_params[p.name] = p.numpy()

            dy_g_loss = g_loss.numpy()
            dy_d_loss = d_loss.numpy()

        dy_params2 = dict()
        with fluid.dygraph.guard():
            fluid.set_flags({'FLAGS_sort_sum_gradient': True})
            paddle.seed(1)
            paddle.framework.random._manual_program_seed(1)
            discriminator2 = Discriminator()
            generator2 = Generator()
<<<<<<< HEAD
            sgd2 = SGDOptimizer(
                learning_rate=1e-3,
                parameter_list=(
                    discriminator2.parameters() + generator2.parameters()
                ),
            )

            d_real2 = discriminator2(to_variable(np.ones([2, 1], np.float32)))
            d_loss_real2 = paddle.mean(
                paddle.nn.functional.binary_cross_entropy_with_logits(
                    logit=d_real2,
                    label=to_variable(np.ones([2, 1], np.float32)),
                )
            )

            d_fake2 = discriminator2(
                generator2(to_variable(np.ones([2, 2], np.float32)))
            )
            d_loss_fake2 = paddle.mean(
                paddle.nn.functional.binary_cross_entropy_with_logits(
                    logit=d_fake2,
                    label=to_variable(np.zeros([2, 1], np.float32)),
                )
            )
=======
            sgd2 = SGDOptimizer(learning_rate=1e-3,
                                parameter_list=(discriminator2.parameters() +
                                                generator2.parameters()))

            d_real2 = discriminator2(to_variable(np.ones([2, 1], np.float32)))
            d_loss_real2 = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_real2, label=to_variable(np.ones([2, 1], np.float32))))

            d_fake2 = discriminator2(
                generator2(to_variable(np.ones([2, 2], np.float32))))
            d_loss_fake2 = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_fake2, label=to_variable(np.zeros([2, 1], np.float32))))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            d_loss2 = d_loss_real2 + d_loss_fake2
            d_loss2.backward()
            sgd2.minimize(d_loss2)
            discriminator2.clear_gradients()
            generator2.clear_gradients()

            d_fake2 = discriminator2(
<<<<<<< HEAD
                generator2(to_variable(np.ones([2, 2], np.float32)))
            )
            g_loss2 = paddle.mean(
                paddle.nn.functional.binary_cross_entropy_with_logits(
                    logit=d_fake2,
                    label=to_variable(np.ones([2, 1], np.float32)),
                )
            )
=======
                generator2(to_variable(np.ones([2, 2], np.float32))))
            g_loss2 = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_fake2, label=to_variable(np.ones([2, 1], np.float32))))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            g_loss2.backward()
            sgd2.minimize(g_loss2)
            for p in discriminator2.parameters():
                dy_params2[p.name] = p.numpy()
            for p in generator.parameters():
                dy_params2[p.name] = p.numpy()

            dy_g_loss2 = g_loss2.numpy()
            dy_d_loss2 = d_loss2.numpy()

        self.assertEqual(dy_g_loss, static_g_loss)
        self.assertEqual(dy_d_loss, static_d_loss)
<<<<<<< HEAD
        for k, v in dy_params.items():
=======
        for k, v in six.iteritems(dy_params):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            np.testing.assert_allclose(v, static_params[k], rtol=1e-05)

        self.assertEqual(dy_g_loss2, static_g_loss)
        self.assertEqual(dy_d_loss2, static_d_loss)
<<<<<<< HEAD
        for k, v in dy_params2.items():
            np.testing.assert_allclose(v, static_params[k], rtol=1e-05)

=======
        for k, v in six.iteritems(dy_params2):
            np.testing.assert_allclose(v, static_params[k], rtol=1e-05)

    def test_gan_float32(self):
        with _test_eager_guard():
            self.func_test_gan_float32()
        self.func_test_gan_float32()

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

if __name__ == '__main__':
    unittest.main()
