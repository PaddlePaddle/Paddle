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

import paddle
import paddle.fluid as fluid
from paddle import _legacy_C_ops
from paddle.fluid.framework import _test_eager_guard
from paddle.tensor import random

if fluid.is_compiled_with_cuda():
    fluid.core.globals()['FLAGS_cudnn_deterministic'] = True


class Config:
    def __init__(self, place, sort_sum_gradient=True):
        self.place = place

        if isinstance(place, fluid.CPUPlace):
            # CPU cases are extremely slow
            self.g_base_dims = 1
            self.d_base_dims = 1

            self.g_repeat_num = 1
            self.d_repeat_num = 1

            self.image_size = 32
        else:
            self.g_base_dims = 64
            self.d_base_dims = 64

            self.g_repeat_num = 6
            self.d_repeat_num = 6

            self.image_size = 256

        self.c_dim = 10
        self.batch_size = 1

        self.seed = 1

        self.lambda_rec = 10
        self.lambda_gp = 10

        self.iterations = 10

        self.sort_sum_gradient = sort_sum_gradient


def create_mnist_dataset(cfg):
    def create_target_label(label):
        return label
        # return (label + 1) % cfg.c_dim # fake label target

    def create_one_hot(label):
        ret = np.zeros([cfg.c_dim])
        ret[label] = 1
        return ret

    def __impl__():
        dataset = paddle.dataset.mnist.train()
        image_reals = []
        label_orgs = []
        label_trgs = []
        num = 0

        for image_real, label_org in dataset():
            image_real = np.reshape(np.array(image_real), [28, 28])
            image_real = np.resize(image_real, [cfg.image_size, cfg.image_size])
            image_real = np.array([image_real] * 3)

            label_trg = create_target_label(label_org)

            image_reals.append(np.array(image_real))
            label_orgs.append(create_one_hot(label_org))
            label_trgs.append(create_one_hot(label_trg))

            if len(image_reals) == cfg.batch_size:
                image_real_np = np.array(image_reals).astype('float32')
                label_org_np = np.array(label_orgs).astype('float32')
                label_trg_np = np.array(label_trgs).astype('float32')

                yield image_real_np, label_org_np, label_trg_np

                num += 1
                if num == cfg.iterations:
                    break

                image_reals = []
                label_orgs = []
                label_trgs = []

    return __impl__


class InstanceNorm(fluid.dygraph.Layer):
    def __init__(self, num_channels, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.scale = self.create_parameter(shape=[num_channels], is_bias=False)
        self.bias = self.create_parameter(shape=[num_channels], is_bias=True)

    def forward(self, input):
        if fluid._non_static_mode():
            out, _, _ = _legacy_C_ops.instance_norm(
                input, self.scale, self.bias, 'epsilon', self.epsilon
            )
            return out
        else:
            return paddle.static.nn.instance_norm(
                input,
                epsilon=self.epsilon,
                param_attr=fluid.ParamAttr(self.scale.name),
                bias_attr=fluid.ParamAttr(self.bias.name),
            )


class Conv2DLayer(fluid.dygraph.Layer):
    def __init__(
        self,
        num_channels,
        num_filters=64,
        filter_size=7,
        stride=1,
        padding=0,
        norm=None,
        use_bias=False,
        relufactor=None,
    ):
        super().__init__()
        self._conv = paddle.nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            bias_attr=None if use_bias else False,
        )

        if norm is not None:
            self._norm = InstanceNorm(num_filters)
        else:
            self._norm = None

        self.relufactor = relufactor

    def forward(self, input):
        conv = self._conv(input)

        if self._norm:
            conv = self._norm(conv)

        if self.relufactor is not None:
            conv = paddle.nn.functional.leaky_relu(conv, self.relufactor)

        return conv


class Deconv2DLayer(fluid.dygraph.Layer):
    def __init__(
        self,
        num_channels,
        num_filters=64,
        filter_size=7,
        stride=1,
        padding=0,
        norm=None,
        use_bias=False,
        relufactor=None,
    ):
        super().__init__()

        self._deconv = paddle.nn.Conv2DTranspose(
            num_channels,
            num_filters,
            filter_size,
            stride=stride,
            padding=padding,
            bias_attr=None if use_bias else False,
        )

        if norm is not None:
            self._norm = InstanceNorm(num_filters)
        else:
            self._norm = None

        self.relufactor = relufactor

    def forward(self, input):
        deconv = self._deconv(input)

        if self._norm:
            deconv = self._norm(deconv)

        if self.relufactor is not None:
            deconv = paddle.nn.functional.leaky_relu(deconv, self.relufactor)

        return deconv


class ResidualBlock(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters):
        super().__init__()
        self._conv0 = Conv2DLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=3,
            stride=1,
            padding=1,
            norm=True,
            relufactor=0,
        )

        self._conv1 = Conv2DLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=1,
            padding=1,
            norm=True,
            relufactor=None,
        )

    def forward(self, input):
        conv0 = self._conv0(input)
        conv1 = self._conv1(conv0)
        return input + conv1


class Generator(fluid.dygraph.Layer):
    def __init__(self, cfg, num_channels=3):
        super().__init__()
        conv_base = Conv2DLayer(
            num_channels=cfg.c_dim + num_channels,
            num_filters=cfg.g_base_dims,
            filter_size=7,
            stride=1,
            padding=3,
            norm=True,
            relufactor=0,
        )

        sub_layers = [conv_base]
        cur_channels = cfg.g_base_dims
        for i in range(2):
            sub_layer = Conv2DLayer(
                num_channels=cur_channels,
                num_filters=cur_channels * 2,
                filter_size=4,
                stride=2,
                padding=1,
                norm=True,
                relufactor=0,
            )

            cur_channels *= 2
            sub_layers.append(sub_layer)

        self._conv0 = paddle.nn.Sequential(*sub_layers)

        repeat_num = cfg.g_repeat_num
        sub_layers = []
        for i in range(repeat_num):
            res_block = ResidualBlock(
                num_channels=cur_channels, num_filters=cfg.g_base_dims * 4
            )
            sub_layers.append(res_block)

        self._res_block = paddle.nn.Sequential(*sub_layers)

        cur_channels = cfg.g_base_dims * 4
        sub_layers = []
        for i in range(2):
            rate = 2 ** (1 - i)
            deconv = Deconv2DLayer(
                num_channels=cur_channels,
                num_filters=cfg.g_base_dims * rate,
                filter_size=4,
                stride=2,
                padding=1,
                relufactor=0,
                norm=True,
            )
            cur_channels = cfg.g_base_dims * rate
            sub_layers.append(deconv)

        self._deconv = paddle.nn.Sequential(*sub_layers)

        self._conv1 = Conv2DLayer(
            num_channels=cur_channels,
            num_filters=3,
            filter_size=7,
            stride=1,
            padding=3,
            relufactor=None,
        )

    def forward(self, input, label_trg):
        shape = input.shape
        label_trg_e = paddle.reshape(label_trg, [-1, label_trg.shape[1], 1, 1])
        label_trg_e = paddle.expand(label_trg_e, [-1, -1, shape[2], shape[3]])

        input1 = fluid.layers.concat([input, label_trg_e], 1)

        conv0 = self._conv0(input1)
        res_block = self._res_block(conv0)
        deconv = self._deconv(res_block)
        conv1 = self._conv1(deconv)
        out = paddle.tanh(conv1)
        return out


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, cfg, num_channels=3):
        super().__init__()

        cur_dim = cfg.d_base_dims

        conv_base = Conv2DLayer(
            num_channels=num_channels,
            num_filters=cur_dim,
            filter_size=4,
            stride=2,
            padding=1,
            relufactor=0.2,
        )

        repeat_num = cfg.d_repeat_num
        sub_layers = [conv_base]
        for i in range(1, repeat_num):
            sub_layer = Conv2DLayer(
                num_channels=cur_dim,
                num_filters=cur_dim * 2,
                filter_size=4,
                stride=2,
                padding=1,
                relufactor=0.2,
            )
            cur_dim *= 2
            sub_layers.append(sub_layer)

        self._conv0 = paddle.nn.Sequential(*sub_layers)

        kernel_size = int(cfg.image_size / np.power(2, repeat_num))

        self._conv1 = Conv2DLayer(
            num_channels=cur_dim,
            num_filters=1,
            filter_size=3,
            stride=1,
            padding=1,
        )

        self._conv2 = Conv2DLayer(
            num_channels=cur_dim, num_filters=cfg.c_dim, filter_size=kernel_size
        )

    def forward(self, input):
        conv = self._conv0(input)
        out1 = self._conv1(conv)
        out2 = self._conv2(conv)
        return out1, out2


def loss_cls(cls, label, cfg):
    cls_shape = cls.shape
    cls = paddle.reshape(cls, [-1, cls_shape[1] * cls_shape[2] * cls_shape[3]])
    return (
        paddle.sum(
            paddle.nn.functional.binary_cross_entropy_with_logits(cls, label)
        )
        / cfg.batch_size
    )


def calc_gradients(outputs, inputs, no_grad_set):
    if fluid._non_static_mode():
        return fluid.dygraph.grad(
            outputs=outputs,
            inputs=inputs,
            no_grad_vars=no_grad_set,
            create_graph=True,
        )
    else:
        return fluid.gradients(
            targets=outputs, inputs=inputs, no_grad_set=no_grad_set
        )


def gradient_penalty(f, real, fake, no_grad_set, cfg):
    def _interpolate(a, b):
        shape = [a.shape[0]]
        alpha = random.uniform_random_batch_size_like(
            input=a, shape=shape, min=0.1, max=1.0, seed=cfg.seed
        )

        inner = paddle.tensor.math._multiply_with_axis(
            b, 1.0 - alpha, axis=0
        ) + paddle.tensor.math._multiply_with_axis(a, alpha, axis=0)
        return inner

    x = _interpolate(real, fake)
    pred, _ = f(x)
    if isinstance(pred, tuple):
        pred = pred[0]

    gradient = calc_gradients(
        outputs=[pred], inputs=[x], no_grad_set=no_grad_set
    )

    if gradient is None:
        return None

    gradient = gradient[0]
    grad_shape = gradient.shape

    gradient = paddle.reshape(
        gradient, [-1, grad_shape[1] * grad_shape[2] * grad_shape[3]]
    )

    epsilon = 1e-16
    norm = paddle.sqrt(paddle.sum(paddle.square(gradient), axis=1) + epsilon)

    gp = paddle.mean(paddle.square(norm - 1.0))
    return gp


def get_generator_loss(
    image_real, label_org, label_trg, generator, discriminator, cfg
):
    fake_img = generator(image_real, label_trg)
    rec_img = generator(fake_img, label_org)
    g_loss_rec = paddle.mean(paddle.abs(paddle.subtract(image_real, rec_img)))

    pred_fake, cls_fake = discriminator(fake_img)

    g_loss_fake = -paddle.mean(pred_fake)
    g_loss_cls = loss_cls(cls_fake, label_trg, cfg)
    g_loss = g_loss_fake + cfg.lambda_rec * g_loss_rec + g_loss_cls
    return g_loss


def get_discriminator_loss(
    image_real, label_org, label_trg, generator, discriminator, cfg
):
    fake_img = generator(image_real, label_trg)
    pred_real, cls_real = discriminator(image_real)
    pred_fake, _ = discriminator(fake_img)
    d_loss_cls = loss_cls(cls_real, label_org, cfg)
    d_loss_fake = paddle.mean(pred_fake)
    d_loss_real = -paddle.mean(pred_real)
    d_loss = d_loss_real + d_loss_fake + d_loss_cls

    d_loss_gp = gradient_penalty(
        discriminator,
        image_real,
        fake_img,
        set(discriminator.parameters()),
        cfg,
    )
    if d_loss_gp is not None:
        d_loss += cfg.lambda_gp * d_loss_gp

    return d_loss


def build_optimizer(layer, cfg, loss=None):
    learning_rate = 1e-3
    beta1 = 0.5
    beta2 = 0.999
    if fluid._non_static_mode():
        return fluid.optimizer.Adam(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            parameter_list=layer.parameters(),
        )
    else:
        optimizer = fluid.optimizer.Adam(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2
        )

        optimizer.minimize(loss, parameter_list=layer.parameters())
        return optimizer


class DyGraphTrainModel:
    def __init__(self, cfg):
        paddle.seed(1)
        paddle.framework.random._manual_program_seed(1)

        self.generator = Generator(cfg)
        self.discriminator = Discriminator(cfg)

        self.g_optimizer = build_optimizer(self.generator, cfg)
        self.d_optimizer = build_optimizer(self.discriminator, cfg)

        self.cfg = cfg

        fluid.set_flags({'FLAGS_sort_sum_gradient': cfg.sort_sum_gradient})

    def clear_gradients(self):
        if self.g_optimizer:
            self.g_optimizer.clear_gradients()

        if self.d_optimizer:
            self.d_optimizer.clear_gradients()

    def run(self, image_real, label_org, label_trg):
        image_real = fluid.dygraph.to_variable(image_real)
        label_org = fluid.dygraph.to_variable(label_org)
        label_trg = fluid.dygraph.to_variable(label_trg)

        g_loss = get_generator_loss(
            image_real,
            label_org,
            label_trg,
            self.generator,
            self.discriminator,
            self.cfg,
        )
        g_loss.backward()
        if self.g_optimizer:
            self.g_optimizer.minimize(g_loss)

        self.clear_gradients()

        d_loss = get_discriminator_loss(
            image_real,
            label_org,
            label_trg,
            self.generator,
            self.discriminator,
            self.cfg,
        )
        d_loss.backward()
        if self.d_optimizer:
            self.d_optimizer.minimize(d_loss)

        self.clear_gradients()

        return g_loss.numpy()[0], d_loss.numpy()[0]


class StaticGraphTrainModel:
    def __init__(self, cfg):
        self.cfg = cfg

        def create_data_layer():
            image_real = fluid.data(
                shape=[None, 3, cfg.image_size, cfg.image_size],
                dtype='float32',
                name='image_real',
            )
            label_org = fluid.data(
                shape=[None, cfg.c_dim], dtype='float32', name='label_org'
            )
            label_trg = fluid.data(
                shape=[None, cfg.c_dim], dtype='float32', name='label_trg'
            )
            return image_real, label_org, label_trg

        paddle.seed(cfg.seed)
        paddle.framework.random._manual_program_seed(cfg.seed)
        self.gen_program = fluid.Program()
        gen_startup_program = fluid.Program()

        with fluid.program_guard(self.gen_program, gen_startup_program):
            with fluid.unique_name.guard():
                image_real, label_org, label_trg = create_data_layer()
                generator = Generator(cfg)
                discriminator = Discriminator(cfg)
                g_loss = get_generator_loss(
                    image_real,
                    label_org,
                    label_trg,
                    generator,
                    discriminator,
                    cfg,
                )
                build_optimizer(generator, cfg, loss=g_loss)

        self.dis_program = fluid.Program()
        dis_startup_program = fluid.Program()
        with fluid.program_guard(self.dis_program, dis_startup_program):
            with fluid.unique_name.guard():
                image_real, label_org, label_trg = create_data_layer()
                generator = Generator(cfg)
                discriminator = Discriminator(cfg)
                d_loss = get_discriminator_loss(
                    image_real,
                    label_org,
                    label_trg,
                    generator,
                    discriminator,
                    cfg,
                )
                build_optimizer(discriminator, cfg, loss=d_loss)

        self.executor = fluid.Executor(cfg.place)
        self.scope = fluid.Scope()

        with fluid.scope_guard(self.scope):
            self.executor.run(gen_startup_program)
            self.executor.run(dis_startup_program)

        self.g_loss = g_loss
        self.d_loss = d_loss

    def run(self, image_real, label_org, label_trg):
        feed = {
            'image_real': image_real,
            'label_org': label_org,
            'label_trg': label_trg,
        }
        with fluid.scope_guard(self.scope):
            g_loss_val = self.executor.run(
                self.gen_program, feed=feed, fetch_list=[self.g_loss]
            )[0]
            d_loss_val = self.executor.run(
                self.dis_program, feed=feed, fetch_list=[self.d_loss]
            )[0]
            return g_loss_val[0], d_loss_val[0]


class TestStarGANWithGradientPenalty(unittest.TestCase):
    def func_main(self):
        self.place_test(fluid.CPUPlace())

        if fluid.is_compiled_with_cuda():
            self.place_test(fluid.CUDAPlace(0))

    def place_test(self, place):
        cfg = Config(place, False)

        dataset = create_mnist_dataset(cfg)
        dataset = paddle.reader.cache(dataset)

        fluid_dygraph_loss = []
        with fluid.dygraph.guard(cfg.place):
            fluid_dygraph_model = DyGraphTrainModel(cfg)
            for batch_id, (image_real, label_org, label_trg) in enumerate(
                dataset()
            ):
                loss = fluid_dygraph_model.run(image_real, label_org, label_trg)
                fluid_dygraph_loss.append(loss)

        eager_dygraph_loss = []
        with _test_eager_guard():
            with fluid.dygraph.guard(cfg.place):
                eager_dygraph_model = DyGraphTrainModel(cfg)
                for batch_id, (image_real, label_org, label_trg) in enumerate(
                    dataset()
                ):
                    loss = eager_dygraph_model.run(
                        image_real, label_org, label_trg
                    )
                    eager_dygraph_loss.append(loss)

        for (g_loss_f, d_loss_f), (g_loss_e, d_loss_e) in zip(
            fluid_dygraph_loss, eager_dygraph_loss
        ):
            self.assertEqual(g_loss_f, g_loss_e)
            self.assertEqual(d_loss_f, d_loss_e)

    def test_all_cases(self):
        self.func_main()


class TestStarGANWithGradientPenaltyLegacy(unittest.TestCase):
    def func_main(self):
        self.place_test(fluid.CPUPlace())

        if fluid.is_compiled_with_cuda():
            self.place_test(fluid.CUDAPlace(0))

    def place_test(self, place):
        cfg = Config(place)

        dataset = create_mnist_dataset(cfg)
        dataset = paddle.reader.cache(dataset)

        static_graph_model = StaticGraphTrainModel(cfg)
        static_loss = []
        for batch_id, (image_real, label_org, label_trg) in enumerate(
            dataset()
        ):
            loss = static_graph_model.run(image_real, label_org, label_trg)
            static_loss.append(loss)

        dygraph_loss = []
        with fluid.dygraph.guard(cfg.place):
            dygraph_model = DyGraphTrainModel(cfg)
            for batch_id, (image_real, label_org, label_trg) in enumerate(
                dataset()
            ):
                loss = dygraph_model.run(image_real, label_org, label_trg)
                dygraph_loss.append(loss)

        for (g_loss_s, d_loss_s), (g_loss_d, d_loss_d) in zip(
            static_loss, dygraph_loss
        ):
            self.assertEqual(g_loss_s, g_loss_d)
            self.assertEqual(d_loss_s, d_loss_d)

    def test_all_cases(self):
        self.func_main()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
