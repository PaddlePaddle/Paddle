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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle.fluid as fluid
from hapi.model import Model
from hapi.loss import Loss

from layers import ConvBN, DeConvBN


class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, dropout=False):
        super(ResnetBlock, self).__init__()
        self.dropout = dropout
        self.conv0 = ConvBN(dim, dim, 3, 1)
        self.conv1 = ConvBN(dim, dim, 3, 1, act=None)

    def forward(self, inputs):
        out_res = fluid.layers.pad2d(inputs, [1, 1, 1, 1], mode="reflect")
        out_res = self.conv0(out_res)
        if self.dropout:
            out_res = fluid.layers.dropout(out_res, dropout_prob=0.5)
        out_res = fluid.layers.pad2d(out_res, [1, 1, 1, 1], mode="reflect")
        out_res = self.conv1(out_res)
        return out_res + inputs


class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, input_channel, n_blocks=9, dropout=False):
        super(ResnetGenerator, self).__init__()

        self.conv0 = ConvBN(input_channel, 32, 7, 1)
        self.conv1 = ConvBN(32, 64, 3, 2, padding=1)
        self.conv2 = ConvBN(64, 128, 3, 2, padding=1)

        dim = 128
        self.resnet_blocks = []
        for i in range(n_blocks):
            block = self.add_sublayer("generator_%d" % (i + 1),
                                      ResnetBlock(dim, dropout))
            self.resnet_blocks.append(block)

        self.deconv0 = DeConvBN(
            dim, 32 * 2, 3, 2, padding=[1, 1], outpadding=[0, 1, 0, 1])
        self.deconv1 = DeConvBN(
            32 * 2, 32, 3, 2, padding=[1, 1], outpadding=[0, 1, 0, 1])

        self.conv3 = ConvBN(
            32, input_channel, 7, 1, norm=False, act=False, use_bias=True)

    def forward(self, inputs):
        pad_input = fluid.layers.pad2d(inputs, [3, 3, 3, 3], mode="reflect")
        y = self.conv0(pad_input)
        y = self.conv1(y)
        y = self.conv2(y)
        for resnet_block in self.resnet_blocks:
            y = resnet_block(y)
        y = self.deconv0(y)
        y = self.deconv1(y)
        y = fluid.layers.pad2d(y, [3, 3, 3, 3], mode="reflect")
        y = self.conv3(y)
        y = fluid.layers.tanh(y)
        return y


class NLayerDiscriminator(fluid.dygraph.Layer):
    def __init__(self, input_channel, d_dims=64, d_nlayers=3):
        super(NLayerDiscriminator, self).__init__()
        self.conv0 = ConvBN(
            input_channel,
            d_dims,
            4,
            2,
            1,
            norm=False,
            use_bias=True,
            relufactor=0.2)

        nf_mult, nf_mult_prev = 1, 1
        self.conv_layers = []
        for n in range(1, d_nlayers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            conv = self.add_sublayer(
                'discriminator_%d' % (n),
                ConvBN(
                    d_dims * nf_mult_prev,
                    d_dims * nf_mult,
                    4,
                    2,
                    1,
                    relufactor=0.2))
            self.conv_layers.append(conv)

        nf_mult_prev = nf_mult
        nf_mult = min(2**d_nlayers, 8)
        self.conv4 = ConvBN(
            d_dims * nf_mult_prev, d_dims * nf_mult, 4, 1, 1, relufactor=0.2)
        self.conv5 = ConvBN(
            d_dims * nf_mult,
            1,
            4,
            1,
            1,
            norm=False,
            act=None,
            use_bias=True,
            relufactor=0.2)

    def forward(self, inputs):
        y = self.conv0(inputs)
        for conv in self.conv_layers:
            y = conv(y)
        y = self.conv4(y)
        y = self.conv5(y)
        return y


class Generator(Model):
    def __init__(self, input_channel=3):
        super(Generator, self).__init__()
        self.g = ResnetGenerator(input_channel)

    def forward(self, input):
        fake = self.g(input)
        return fake


class GeneratorCombine(Model):
    def __init__(self, g_AB=None, g_BA=None, d_A=None, d_B=None,
                 is_train=True):
        super(GeneratorCombine, self).__init__()
        self.g_AB = g_AB
        self.g_BA = g_BA
        self.is_train = is_train
        if self.is_train:
            self.d_A = d_A
            self.d_B = d_B

    def forward(self, input_A, input_B):
        # Translate images to the other domain
        fake_B = self.g_AB(input_A)
        fake_A = self.g_BA(input_B)

        # Translate images back to original domain
        cyc_A = self.g_BA(fake_B)
        cyc_B = self.g_AB(fake_A)
        if not self.is_train:
            return fake_A, fake_B, cyc_A, cyc_B

        # Identity mapping of images
        idt_A = self.g_AB(input_B)
        idt_B = self.g_BA(input_A)

        # Discriminators determines validity of translated images
        # d_A(g_AB(A))
        valid_A = self.d_A.d(fake_B)
        # d_B(g_BA(A))
        valid_B = self.d_B.d(fake_A)
        return input_A, input_B, fake_A, fake_B, cyc_A, cyc_B, idt_A, idt_B, valid_A, valid_B


class GLoss(Loss):
    def __init__(self, lambda_A=10., lambda_B=10., lambda_identity=0.5):
        super(GLoss, self).__init__()
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_identity = lambda_identity

    def forward(self, outputs, labels=None):
        input_A, input_B, fake_A, fake_B, cyc_A, cyc_B, idt_A, idt_B, valid_A, valid_B = outputs

        def mse(a, b):
            return fluid.layers.reduce_mean(fluid.layers.square(a - b))

        def mae(a, b):  # L1Loss
            return fluid.layers.reduce_mean(fluid.layers.abs(a - b))

        g_A_loss = mse(valid_A, 1.)
        g_B_loss = mse(valid_B, 1.)
        g_loss = g_A_loss + g_B_loss

        cyc_A_loss = mae(input_A, cyc_A) * self.lambda_A
        cyc_B_loss = mae(input_B, cyc_B) * self.lambda_B
        cyc_loss = cyc_A_loss + cyc_B_loss

        idt_loss_A = mae(input_B, idt_A) * (self.lambda_B *
                                            self.lambda_identity)
        idt_loss_B = mae(input_A, idt_B) * (self.lambda_A *
                                            self.lambda_identity)
        idt_loss = idt_loss_A + idt_loss_B

        loss = cyc_loss + g_loss + idt_loss
        return loss


class Discriminator(Model):
    def __init__(self, input_channel=3):
        super(Discriminator, self).__init__()
        self.d = NLayerDiscriminator(input_channel)

    def forward(self, real, fake):
        pred_real = self.d(real)
        pred_fake = self.d(fake)
        return pred_real, pred_fake


class DLoss(Loss):
    def __init__(self):
        super(DLoss, self).__init__()

    def forward(self, inputs, labels=None):
        pred_real, pred_fake = inputs
        loss = fluid.layers.square(pred_fake) + fluid.layers.square(pred_real -
                                                                    1.)
        loss = fluid.layers.reduce_mean(loss / 2.0)
        return loss
