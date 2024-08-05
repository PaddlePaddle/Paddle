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

import unittest

import numpy as np
from test_imperative_base import new_program_scope

import paddle
from paddle import base
from paddle.base import core
from paddle.nn import Linear


class Discriminator(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._fc1 = Linear(1, 32)
        self._fc2 = Linear(32, 1)

    def forward(self, inputs):
        x = self._fc1(inputs)
        x = paddle.nn.functional.elu(x)
        x = self._fc2(x)
        return x


class Generator(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._fc1 = Linear(2, 64)
        self._fc2 = Linear(64, 64)
        self._fc3 = Linear(64, 1)

    def forward(self, inputs):
        x = self._fc1(inputs)
        x = paddle.nn.functional.elu(x)
        x = self._fc2(x)
        x = paddle.nn.functional.elu(x)
        x = self._fc3(x)
        return x


class TestDygraphGAN(unittest.TestCase):
    def test_gan_float32(self):
        seed = 90
        paddle.seed(1)
        paddle.framework.random._manual_program_seed(1)
        startup = base.Program()
        discriminate_p = base.Program()
        generate_p = base.Program()

        scope = base.core.Scope()
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
                    label=paddle.tensor.fill_constant(
                        shape=[2, 1], dtype='float32', value=1.0
                    ),
                )
            )

            d_fake = discriminator(generator(noise))
            d_loss_fake = paddle.mean(
                paddle.nn.functional.binary_cross_entropy_with_logits(
                    logit=d_fake,
                    label=paddle.tensor.fill_constant(
                        shape=[2, 1], dtype='float32', value=0.0
                    ),
                )
            )

            d_loss = d_loss_real + d_loss_fake

            sgd = paddle.optimizer.SGD(learning_rate=1e-3)
            sgd.minimize(d_loss)

        with new_program_scope(main=generate_p, startup=startup, scope=scope):
            discriminator = Discriminator()
            generator = Generator()

            noise = paddle.static.data(name="noise", shape=[2, 2])

            d_fake = discriminator(generator(noise))
            g_loss = paddle.mean(
                paddle.nn.functional.binary_cross_entropy_with_logits(
                    logit=d_fake,
                    label=paddle.tensor.fill_constant(
                        shape=[2, 1], dtype='float32', value=1.0
                    ),
                )
            )

            sgd = paddle.optimizer.SGD(learning_rate=1e-3)
            sgd.minimize(g_loss)

        exe = base.Executor(
            base.CPUPlace()
            if not core.is_compiled_with_cuda()
            else base.CUDAPlace(0)
        )
        static_params = {}
        with base.scope_guard(scope):
            img = np.ones([2, 1], np.float32)
            noise = np.ones([2, 2], np.float32)
            exe.run(startup)
            static_d_loss = exe.run(
                discriminate_p,
                feed={'img': img, 'noise': noise},
                fetch_list=[d_loss],
            )[0]
            static_g_loss = exe.run(
                generate_p, feed={'noise': noise}, fetch_list=[g_loss]
            )[0]

            # generate_p contains all parameters needed.
            for param in generate_p.global_block().all_parameters():
                static_params[param.name] = np.array(
                    scope.find_var(param.name).get_tensor()
                )

        dy_params = {}
        with base.dygraph.guard():
            paddle.seed(1)
            with paddle.pir_utils.OldIrGuard():
                # Note: dygraph use self.main_program.global_block().create_parameter(), it's need manual seed to old Program
                paddle.framework.random._manual_program_seed(1)

            discriminator = Discriminator()
            generator = Generator()
            sgd = paddle.optimizer.SGD(
                learning_rate=1e-3,
                parameters=(
                    discriminator.parameters() + generator.parameters()
                ),
            )

            d_real = discriminator(
                paddle.to_tensor(np.ones([2, 1], np.float32))
            )
            d_loss_real = paddle.mean(
                paddle.nn.functional.binary_cross_entropy_with_logits(
                    logit=d_real,
                    label=paddle.to_tensor(np.ones([2, 1], np.float32)),
                )
            )

            d_fake = discriminator(
                generator(paddle.to_tensor(np.ones([2, 2], np.float32)))
            )
            d_loss_fake = paddle.mean(
                paddle.nn.functional.binary_cross_entropy_with_logits(
                    logit=d_fake,
                    label=paddle.to_tensor(np.zeros([2, 1], np.float32)),
                )
            )

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            sgd.minimize(d_loss)
            discriminator.clear_gradients()
            generator.clear_gradients()

            d_fake = discriminator(
                generator(paddle.to_tensor(np.ones([2, 2], np.float32)))
            )
            g_loss = paddle.mean(
                paddle.nn.functional.binary_cross_entropy_with_logits(
                    logit=d_fake,
                    label=paddle.to_tensor(np.ones([2, 1], np.float32)),
                )
            )
            g_loss.backward()
            sgd.minimize(g_loss)
            for p in discriminator.parameters():
                dy_params[p.name] = p.numpy()
            for p in generator.parameters():
                dy_params[p.name] = p.numpy()

            dy_g_loss = g_loss.numpy()
            dy_d_loss = d_loss.numpy()

        dy_params2 = {}
        with base.dygraph.guard():
            base.set_flags({'FLAGS_sort_sum_gradient': True})
            paddle.seed(1)
            paddle.framework.random._manual_program_seed(1)
            discriminator2 = Discriminator()
            generator2 = Generator()
            sgd2 = paddle.optimizer.SGD(
                learning_rate=1e-3,
                parameters=(
                    discriminator2.parameters() + generator2.parameters()
                ),
            )

            d_real2 = discriminator2(
                paddle.to_tensor(np.ones([2, 1], np.float32))
            )
            d_loss_real2 = paddle.mean(
                paddle.nn.functional.binary_cross_entropy_with_logits(
                    logit=d_real2,
                    label=paddle.to_tensor(np.ones([2, 1], np.float32)),
                )
            )

            d_fake2 = discriminator2(
                generator2(paddle.to_tensor(np.ones([2, 2], np.float32)))
            )
            d_loss_fake2 = paddle.mean(
                paddle.nn.functional.binary_cross_entropy_with_logits(
                    logit=d_fake2,
                    label=paddle.to_tensor(np.zeros([2, 1], np.float32)),
                )
            )

            d_loss2 = d_loss_real2 + d_loss_fake2
            d_loss2.backward()
            sgd2.minimize(d_loss2)
            discriminator2.clear_gradients()
            generator2.clear_gradients()

            d_fake2 = discriminator2(
                generator2(paddle.to_tensor(np.ones([2, 2], np.float32)))
            )
            g_loss2 = paddle.mean(
                paddle.nn.functional.binary_cross_entropy_with_logits(
                    logit=d_fake2,
                    label=paddle.to_tensor(np.ones([2, 1], np.float32)),
                )
            )
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
        for k, v in dy_params.items():
            np.testing.assert_allclose(v, static_params[k], rtol=1e-05)

        self.assertEqual(dy_g_loss2, static_g_loss)
        self.assertEqual(dy_d_loss2, static_d_loss)
        for k, v in dy_params2.items():
            np.testing.assert_allclose(v, static_params[k], rtol=1e-05)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
