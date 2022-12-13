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

# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import os
import random
import time
import unittest

import numpy as np
from PIL import Image, ImageOps

import paddle.fluid as fluid

# Use GPU:0 to elimate the influence of other tasks.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import paddle
from paddle.fluid.dygraph import to_variable
from paddle.jit import ProgramTranslator
from paddle.jit.api import declarative
from paddle.nn import BatchNorm

# Note: Set True to eliminate randomness.
#     1. For one operation, cuDNN has several algorithms,
#        some algorithm results are non-deterministic, like convolution algorithms.
#     2. If include BatchNorm, please set `use_global_stats=True` to avoid using
#        cudnnBatchNormalizationBackward which is non-deterministic.
if fluid.is_compiled_with_cuda():
    fluid.set_flags({'FLAGS_cudnn_deterministic': True})

# set False to speed up training.
use_cudnn = False
step_per_epoch = 10
lambda_A = 10.0
lambda_B = 10.0
lambda_identity = 0.5
# TODO(Aurelius84): Modify it into 256 when we move ut into CE platform.
# It will lead to timeout if set 256 in CI.
IMAGE_SIZE = 64
SEED = 2020

program_translator = ProgramTranslator()


class Cycle_Gan(fluid.dygraph.Layer):
    def __init__(self, input_channel, istrain=True):
        super().__init__()

        self.build_generator_resnet_9blocks_a = build_generator_resnet_9blocks(
            input_channel
        )
        self.build_generator_resnet_9blocks_b = build_generator_resnet_9blocks(
            input_channel
        )
        if istrain:
            self.build_gen_discriminator_a = build_gen_discriminator(
                input_channel
            )
            self.build_gen_discriminator_b = build_gen_discriminator(
                input_channel
            )

    @declarative
    def forward(self, input_A, input_B):
        """
        Generator of GAN model.
        """
        fake_B = self.build_generator_resnet_9blocks_a(input_A)
        fake_A = self.build_generator_resnet_9blocks_b(input_B)
        cyc_A = self.build_generator_resnet_9blocks_b(fake_B)
        cyc_B = self.build_generator_resnet_9blocks_a(fake_A)

        diff_A = paddle.abs(paddle.subtract(x=input_A, y=cyc_A))
        diff_B = paddle.abs(paddle.subtract(x=input_B, y=cyc_B))
        cyc_A_loss = paddle.mean(diff_A) * lambda_A
        cyc_B_loss = paddle.mean(diff_B) * lambda_B
        cyc_loss = cyc_A_loss + cyc_B_loss

        fake_rec_A = self.build_gen_discriminator_a(fake_B)
        g_A_loss = paddle.mean(paddle.square(fake_rec_A - 1))

        fake_rec_B = self.build_gen_discriminator_b(fake_A)
        g_B_loss = paddle.mean(paddle.square(fake_rec_B - 1))
        G = g_A_loss + g_B_loss
        idt_A = self.build_generator_resnet_9blocks_a(input_B)
        idt_loss_A = (
            paddle.mean(paddle.abs(paddle.subtract(x=input_B, y=idt_A)))
            * lambda_B
            * lambda_identity
        )

        idt_B = self.build_generator_resnet_9blocks_b(input_A)
        idt_loss_B = (
            paddle.mean(paddle.abs(paddle.subtract(x=input_A, y=idt_B)))
            * lambda_A
            * lambda_identity
        )
        idt_loss = paddle.add(idt_loss_A, idt_loss_B)
        g_loss = cyc_loss + G + idt_loss
        return (
            fake_A,
            fake_B,
            cyc_A,
            cyc_B,
            g_A_loss,
            g_B_loss,
            idt_loss_A,
            idt_loss_B,
            cyc_A_loss,
            cyc_B_loss,
            g_loss,
        )

    @declarative
    def discriminatorA(self, input_A, input_B):
        """
        Discriminator A of GAN model.
        """
        rec_B = self.build_gen_discriminator_a(input_A)
        fake_pool_rec_B = self.build_gen_discriminator_a(input_B)

        return rec_B, fake_pool_rec_B

    @declarative
    def discriminatorB(self, input_A, input_B):
        """
        Discriminator B of GAN model.
        """
        rec_A = self.build_gen_discriminator_b(input_A)
        fake_pool_rec_A = self.build_gen_discriminator_b(input_B)

        return rec_A, fake_pool_rec_A


class build_resnet_block(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias=False):
        super().__init__()

        self.conv0 = conv2d(
            num_channels=dim,
            num_filters=dim,
            filter_size=3,
            stride=1,
            stddev=0.02,
            use_bias=False,
        )
        self.conv1 = conv2d(
            num_channels=dim,
            num_filters=dim,
            filter_size=3,
            stride=1,
            stddev=0.02,
            relu=False,
            use_bias=False,
        )
        self.dim = dim

    def forward(self, inputs):
        pad1 = paddle.nn.Pad2D([1, 1, 1, 1], mode="reflect")
        out_res = pad1(inputs)
        out_res = self.conv0(out_res)

        pad2 = paddle.nn.Pad2D([1, 1, 1, 1], mode="reflect")
        out_res = pad2(out_res)
        out_res = self.conv1(out_res)
        return out_res + inputs


class build_generator_resnet_9blocks(fluid.dygraph.Layer):
    def __init__(self, input_channel):
        super().__init__()

        self.conv0 = conv2d(
            num_channels=input_channel,
            num_filters=32,
            filter_size=7,
            stride=1,
            padding=0,
            stddev=0.02,
        )
        self.conv1 = conv2d(
            num_channels=32,
            num_filters=64,
            filter_size=3,
            stride=2,
            padding=1,
            stddev=0.02,
        )
        self.conv2 = conv2d(
            num_channels=64,
            num_filters=128,
            filter_size=3,
            stride=2,
            padding=1,
            stddev=0.02,
        )
        self.build_resnet_block_list = []
        dim = 128
        for i in range(9):
            Build_Resnet_Block = self.add_sublayer(
                "generator_%d" % (i + 1), build_resnet_block(dim)
            )
            self.build_resnet_block_list.append(Build_Resnet_Block)
        self.deconv0 = DeConv2D(
            num_channels=dim,
            num_filters=32 * 2,
            filter_size=3,
            stride=2,
            stddev=0.02,
            padding=[1, 1],
            outpadding=[0, 1, 0, 1],
        )
        self.deconv1 = DeConv2D(
            num_channels=32 * 2,
            num_filters=32,
            filter_size=3,
            stride=2,
            stddev=0.02,
            padding=[1, 1],
            outpadding=[0, 1, 0, 1],
        )
        self.conv3 = conv2d(
            num_channels=32,
            num_filters=input_channel,
            filter_size=7,
            stride=1,
            stddev=0.02,
            padding=0,
            relu=False,
            norm=False,
            use_bias=True,
        )

    def forward(self, inputs):
        pad1 = paddle.nn.Pad2D([3, 3, 3, 3], mode="reflect")
        pad_input = pad1(inputs)
        y = self.conv0(pad_input)
        y = self.conv1(y)
        y = self.conv2(y)
        for build_resnet_block_i in self.build_resnet_block_list:
            y = build_resnet_block_i(y)
        y = self.deconv0(y)
        y = self.deconv1(y)
        pad2 = paddle.nn.Pad2D([3, 3, 3, 3], mode="reflect")
        y = pad2(y)
        y = self.conv3(y)
        y = paddle.tanh(y)
        return y


class build_gen_discriminator(fluid.dygraph.Layer):
    def __init__(self, input_channel):
        super().__init__()

        self.conv0 = conv2d(
            num_channels=input_channel,
            num_filters=64,
            filter_size=4,
            stride=2,
            stddev=0.02,
            padding=1,
            norm=False,
            use_bias=True,
            relufactor=0.2,
        )
        self.conv1 = conv2d(
            num_channels=64,
            num_filters=128,
            filter_size=4,
            stride=2,
            stddev=0.02,
            padding=1,
            relufactor=0.2,
        )
        self.conv2 = conv2d(
            num_channels=128,
            num_filters=IMAGE_SIZE,
            filter_size=4,
            stride=2,
            stddev=0.02,
            padding=1,
            relufactor=0.2,
        )
        self.conv3 = conv2d(
            num_channels=IMAGE_SIZE,
            num_filters=512,
            filter_size=4,
            stride=1,
            stddev=0.02,
            padding=1,
            relufactor=0.2,
        )
        self.conv4 = conv2d(
            num_channels=512,
            num_filters=1,
            filter_size=4,
            stride=1,
            stddev=0.02,
            padding=1,
            norm=False,
            relu=False,
            use_bias=True,
        )

    def forward(self, inputs):
        y = self.conv0(inputs)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        return y


class conv2d(fluid.dygraph.Layer):
    """docstring for Conv2D"""

    def __init__(
        self,
        num_channels,
        num_filters=64,
        filter_size=7,
        stride=1,
        stddev=0.02,
        padding=0,
        norm=True,
        relu=True,
        relufactor=0.0,
        use_bias=False,
    ):
        super().__init__()

        if not use_bias:
            con_bias_attr = False
        else:
            con_bias_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.0)
            )

        self.conv = paddle.nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            weight_attr=paddle.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=stddev
                )
            ),
            bias_attr=con_bias_attr,
        )
        # Note(Aurelius84): The calculation of GPU kernel in BN is non-deterministic,
        # failure rate is 1/100 in Dev but seems incremental in CE platform.
        # If on GPU, we disable BN temporarily.
        if fluid.is_compiled_with_cuda():
            norm = False
        if norm:
            self.bn = BatchNorm(
                use_global_stats=True,  # set True to use deterministic algorithm
                num_channels=num_filters,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.NormalInitializer(1.0, 0.02)
                ),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(0.0)
                ),
                trainable_statistics=True,
            )

        self.relufactor = relufactor
        self.use_bias = use_bias
        self.norm = norm
        self.relu = relu

    def forward(self, inputs):
        conv = self.conv(inputs)
        if self.norm:
            conv = self.bn(conv)
        if self.relu:
            conv = paddle.nn.functional.leaky_relu(conv, self.relufactor)
        return conv


class DeConv2D(fluid.dygraph.Layer):
    def __init__(
        self,
        num_channels,
        num_filters=64,
        filter_size=7,
        stride=1,
        stddev=0.02,
        padding=[0, 0],
        outpadding=[0, 0, 0, 0],
        relu=True,
        norm=True,
        relufactor=0.0,
        use_bias=False,
    ):
        super().__init__()

        if not use_bias:
            de_bias_attr = False
        else:
            de_bias_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.0)
            )

        self._deconv = paddle.nn.Conv2DTranspose(
            num_channels,
            num_filters,
            filter_size,
            stride=stride,
            padding=padding,
            weight_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=stddev
                )
            ),
            bias_attr=de_bias_attr,
        )
        if fluid.is_compiled_with_cuda():
            norm = False
        if norm:
            self.bn = BatchNorm(
                use_global_stats=True,  # set True to use deterministic algorithm
                num_channels=num_filters,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.NormalInitializer(1.0, 0.02)
                ),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(0.0)
                ),
                trainable_statistics=True,
            )

        self.outpadding = outpadding
        self.relufactor = relufactor
        self.use_bias = use_bias
        self.norm = norm
        self.relu = relu

    def forward(self, inputs):
        conv = self._deconv(inputs)
        tmp_pad = paddle.nn.Pad2D(
            padding=self.outpadding, mode='constant', value=0.0
        )
        conv = tmp_pad(conv)

        if self.norm:
            conv = self.bn(conv)
        if self.relu:
            conv = paddle.nn.functional.leaky_relu(conv, self.relufactor)
        return conv


class ImagePool:
    def __init__(self, pool_size=50):
        self.pool = []
        self.count = 0
        self.pool_size = pool_size

    def pool_image(self, image):
        if self.count < self.pool_size:
            self.pool.append(image)
            self.count += 1
            return image
        else:
            p = np.random.rand()
            if p > 0.5:
                random_id = np.random.randint(0, self.pool_size - 1)
                temp = self.pool[random_id]
                self.pool[random_id] = image
                return temp
            else:
                return image


def reader_creater():
    def reader():
        while True:
            fake_image = np.uint8(
                np.random.random((IMAGE_SIZE + 30, IMAGE_SIZE + 30, 3)) * 255
            )
            image = Image.fromarray(fake_image)
            # Resize
            image = image.resize((286, 286), Image.BICUBIC)
            # RandomCrop
            i = np.random.randint(0, 30)
            j = np.random.randint(0, 30)
            image = image.crop((i, j, i + IMAGE_SIZE, j + IMAGE_SIZE))
            # RandomHorizontalFlip
            sed = np.random.rand()
            if sed > 0.5:
                image = ImageOps.mirror(image)
            # ToTensor
            image = np.array(image).transpose([2, 0, 1]).astype('float32')
            image = image / 255.0
            # Normalize, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
            image = (image - 0.5) / 0.5

            yield image

    return reader


class Args:
    epoch = 1
    batch_size = 4
    image_shape = [3, IMAGE_SIZE, IMAGE_SIZE]
    max_images_num = step_per_epoch
    log_step = 1
    train_step = 3


def optimizer_setting(parameters):
    lr = 0.0002
    optimizer = fluid.optimizer.Adam(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=[
                100 * step_per_epoch,
                120 * step_per_epoch,
                140 * step_per_epoch,
                160 * step_per_epoch,
                180 * step_per_epoch,
            ],
            values=[lr, lr * 0.8, lr * 0.6, lr * 0.4, lr * 0.2, lr * 0.1],
        ),
        parameter_list=parameters,
        beta1=0.5,
    )
    return optimizer


def train(args, to_static):
    place = (
        fluid.CUDAPlace(0)
        if fluid.is_compiled_with_cuda()
        else fluid.CPUPlace()
    )

    program_translator.enable(to_static)

    with fluid.dygraph.guard(place):
        max_images_num = args.max_images_num
        data_shape = [-1] + args.image_shape

        random.seed(SEED)
        np.random.seed(SEED)
        fluid.default_startup_program().random_seed = SEED
        fluid.default_main_program().random_seed = SEED

        A_pool = ImagePool()
        B_pool = ImagePool()
        A_reader = paddle.batch(reader_creater(), args.batch_size)()
        B_reader = paddle.batch(reader_creater(), args.batch_size)()
        cycle_gan = Cycle_Gan(input_channel=data_shape[1], istrain=True)

        t_time = 0
        vars_G = (
            cycle_gan.build_generator_resnet_9blocks_a.parameters()
            + cycle_gan.build_generator_resnet_9blocks_b.parameters()
        )
        vars_da = cycle_gan.build_gen_discriminator_a.parameters()
        vars_db = cycle_gan.build_gen_discriminator_b.parameters()

        optimizer1 = optimizer_setting(vars_G)
        optimizer2 = optimizer_setting(vars_da)
        optimizer3 = optimizer_setting(vars_db)

        loss_data = []
        for epoch in range(args.epoch):
            for batch_id in range(max_images_num):

                data_A = next(A_reader)
                data_B = next(B_reader)

                s_time = time.time()
                data_A = np.array(
                    [data_A[0].reshape(3, IMAGE_SIZE, IMAGE_SIZE)]
                ).astype("float32")
                data_B = np.array(
                    [data_B[0].reshape(3, IMAGE_SIZE, IMAGE_SIZE)]
                ).astype("float32")
                data_A = to_variable(data_A)
                data_B = to_variable(data_B)

                # optimize the g_A network
                (
                    fake_A,
                    fake_B,
                    cyc_A,
                    cyc_B,
                    g_A_loss,
                    g_B_loss,
                    idt_loss_A,
                    idt_loss_B,
                    cyc_A_loss,
                    cyc_B_loss,
                    g_loss,
                ) = cycle_gan(data_A, data_B)

                g_loss.backward()
                optimizer1.minimize(g_loss)
                cycle_gan.clear_gradients()

                fake_pool_B = B_pool.pool_image(fake_B).numpy()
                fake_pool_B = np.array(
                    [fake_pool_B[0].reshape(3, IMAGE_SIZE, IMAGE_SIZE)]
                ).astype("float32")
                fake_pool_B = to_variable(fake_pool_B)

                fake_pool_A = A_pool.pool_image(fake_A).numpy()
                fake_pool_A = np.array(
                    [fake_pool_A[0].reshape(3, IMAGE_SIZE, IMAGE_SIZE)]
                ).astype("float32")
                fake_pool_A = to_variable(fake_pool_A)

                # optimize the d_A network
                rec_B, fake_pool_rec_B = cycle_gan.discriminatorA(
                    data_B, fake_pool_B
                )
                d_loss_A = (
                    paddle.square(fake_pool_rec_B) + paddle.square(rec_B - 1)
                ) / 2.0
                d_loss_A = paddle.mean(d_loss_A)

                d_loss_A.backward()
                optimizer2.minimize(d_loss_A)
                cycle_gan.clear_gradients()

                # optimize the d_B network
                rec_A, fake_pool_rec_A = cycle_gan.discriminatorB(
                    data_A, fake_pool_A
                )
                d_loss_B = (
                    paddle.square(fake_pool_rec_A) + paddle.square(rec_A - 1)
                ) / 2.0
                d_loss_B = paddle.mean(d_loss_B)

                d_loss_B.backward()
                optimizer3.minimize(d_loss_B)

                cycle_gan.clear_gradients()

                # Log generator loss and discriminator loss
                cur_batch_loss = [
                    g_loss,
                    d_loss_A,
                    d_loss_B,
                    g_A_loss,
                    cyc_A_loss,
                    idt_loss_A,
                    g_B_loss,
                    cyc_B_loss,
                    idt_loss_B,
                ]
                cur_batch_loss = [x.numpy()[0] for x in cur_batch_loss]

                batch_time = time.time() - s_time
                t_time += batch_time
                if batch_id % args.log_step == 0:
                    print(
                        "batch: {}\t Batch_time_cost: {}\n g_loss: {}\t d_A_loss: {}\t d_B_loss:{}\n g_A_loss: {}\t g_A_cyc_loss: {}\t g_A_idt_loss: {}\n g_B_loss: {}\t g_B_cyc_loss: {}\t g_B_idt_loss: {}".format(
                            batch_id, batch_time, *cur_batch_loss
                        )
                    )

                if batch_id > args.train_step:
                    break

                loss_data.append(cur_batch_loss)
        return np.array(loss_data)


class TestCycleGANModel(unittest.TestCase):
    def setUp(self):
        self.args = Args()

    def train(self, to_static):
        out = train(self.args, to_static)
        return out

    def test_train(self):
        st_out = self.train(to_static=True)
        dy_out = self.train(to_static=False)

        assert_func = np.allclose
        # Note(Aurelius84): Because we disable BN on GPU,
        # but here we enhance the check on CPU by `np.array_equal`
        # which means the dy_out and st_out shall be exactly same.
        if not fluid.is_compiled_with_cuda():
            assert_func = np.array_equal

        self.assertTrue(
            assert_func(dy_out, st_out),
            msg="dy_out:\n {}\n st_out:\n{}".format(dy_out, st_out),
        )


if __name__ == "__main__":
    unittest.main()
