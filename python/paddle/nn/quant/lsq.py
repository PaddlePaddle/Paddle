# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import math

import paddle
from paddle.autograd import PyLayer
from paddle.framework import ParamAttr
from paddle.nn.initializer import Constant
from paddle.utils import unique_name

from ..layer.layers import Layer


def round(x):
    sign = paddle.sign(x)
    x = sign * paddle.floor(paddle.abs(x) + 0.5)
    return x


class LsqFunc(PyLayer):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, per_channel=False, quant_axis=0):
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp, per_channel, quant_axis
        if per_channel:
            sizes = weight.shape
            weight = weight.reshape((weight.shape[quant_axis], -1))
            weight = weight.transpose((1, 0))
            alpha = paddle.broadcast_to(alpha, weight.shape)
            quant_w = round(paddle.divide(weight, alpha)).clip(Qn, Qp)
            quant_w = quant_w * alpha
            quant_w = quant_w.transpose((1, 0))
            quant_w = quant_w.reshape(sizes)
        else:
            quant_w = round(paddle.divide(weight, alpha)).clip(Qn, Qp)
            quant_w = quant_w * alpha
        return quant_w

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensor()
        g, Qn, Qp, per_channel, quant_axis = ctx.other
        if per_channel:
            sizes = weight.shape
            weight = weight.reshape((weight.shape[quant_axis], -1))
            weight = weight.transpose((1, 0))
            alpha = paddle.broadcast_to(alpha, weight.shape)
            q_w = paddle.divide(weight, alpha)
            q_w = q_w.transpose((1, 0))
            q_w = q_w.reshape(sizes)
        else:
            q_w = paddle.divide(weight, alpha)
        lower_flag = paddle.cast((q_w < Qn), 'float32')
        upper_flag = paddle.cast((q_w > Qp), 'float32')
        middle_flag = 1.0 - lower_flag - upper_flag
        if per_channel:
            grad_alpha = (
                (
                    lower_flag * Qn
                    + upper_flag * Qp
                    + middle_flag * round(q_w)
                    - middle_flag * q_w
                )
                * grad_weight
                * g
            )
            grad_alpha = grad_alpha.reshape(
                (grad_alpha.shape[quant_axis], -1)
            ).sum(axis=1)
        else:
            grad_alpha = (
                (
                    (
                        lower_flag * Qn
                        + upper_flag * Qp
                        + middle_flag * round(q_w)
                        - middle_flag * q_w
                    )
                    * grad_weight
                    * g
                )
                .sum()
                .unsqueeze(axis=0)[0]
            )
        grad_weight = middle_flag * grad_weight
        return grad_weight, grad_alpha


class LsqPlusActFunc(PyLayer):
    @staticmethod
    def forward(ctx, x, alpha, beta, g, Qn, Qp):
        ctx.save_for_backward(x, alpha, beta)
        ctx.other = g, Qn, Qp
        quant_x = round(paddle.divide((x - beta), alpha)).clip(Qn, Qp)
        return quant_x * alpha + beta

    @staticmethod
    def backward(ctx, grad_x):
        x, alpha, beta = ctx.saved_tensor()
        g, Qn, Qp = ctx.other
        q_x = (x - beta) / alpha
        lower_flag = paddle.cast((q_x < Qn), 'float32')
        upper_flag = paddle.cast((q_x > Qp), 'float32')
        middle_flag = 1.0 - lower_flag - upper_flag
        grad_alpha = (
            (
                (
                    lower_flag * Qn
                    + upper_flag * Qp
                    + middle_flag * round(q_x)
                    - middle_flag * q_x
                )
                * grad_x
                * g
            )
            .sum()
            .unsqueeze(axis=0)[0]
        )
        grad_beta = (
            ((lower_flag + upper_flag) * grad_x * g).sum().unsqueeze(axis=0)[0]
        )
        grad_x = middle_flag * grad_x
        return grad_x, grad_alpha, grad_beta


class FakeQuantActLSQPlus(Layer):
    def __init__(
        self,
        quant_bits,
        all_positive=False,
        symmetric=False,
        batch_init=20,
        dtype='float32',
        name=None,
        reduce_type=None,
    ):
        super().__init__()
        '''
        Args:
            quant_bits(int): quantization bit number for weights.
            all_positive(bool): whether unsigned or signed quantization, where True for unsigned quantization and False for signed quantization.
            symmetric(bool): whether symmetric or asymmetric quantization.
            batch_init(int): number of batches that collect Gaussian approximation for the weight distribution in each layer.
            dtype(str): data type.
            name(str): the name of the weight.
            reduce_type(str): the reduce type which is needed when parallel training.
        '''
        self.bits = quant_bits
        self.all_positive = all_positive
        self.symmetric = symmetric
        self.batch_init = batch_init
        self.name = name
        self.reduce_type = reduce_type

        if self.all_positive:
            # unsigned activation
            self.Qn = 0
            self.Qp = 2**self.bits - 1
        else:
            # signed activation
            self.Qn = -(2 ** (self.bits - 1))
            self.Qp = 2 ** (self.bits - 1) - 1

        scale_prefix = f"{name}.scale" if name else 'quant_dequant.scale'
        self._scale_name = unique_name.generate(scale_prefix)

        s_attr = ParamAttr(
            name=self._scale_name, initializer=Constant(1.0), trainable=True
        )
        self.s = self.create_parameter(shape=[], attr=s_attr, dtype='float32')
        self.s.stop_gradient = False

        if not self.symmetric:
            beta_prefix = f"{name}.beta" if name else 'quant_dequant.beta'
            self._beta_name = unique_name.generate(beta_prefix)

            beta_attr = ParamAttr(
                name=self._beta_name, initializer=Constant(0.0), trainable=True
            )
            self.beta = self.create_parameter(
                shape=[], attr=beta_attr, dtype='float32'
            )
            self.beta.stop_gradient = False

        self.init_state = 0

    def forward(self, activation):
        if self.reduce_type == "max":
            paddle.distributed.all_reduce(
                self.s, op=paddle.distributed.ReduceOp.MAX
            )

        if not self.symmetric and self.reduce_type == "max":
            paddle.distributed.all_reduce(
                self.beta, op=paddle.distributed.ReduceOp.MAX
            )

        if self.init_state == 0:
            self.g = paddle.to_tensor(
                1.0 / math.sqrt(activation.numel() * self.Qp)
            )
            min_a = paddle.min(activation.detach())
            max_a = paddle.max(activation.detach())
            self.s.set_value((max_a - min_a) / (self.Qp - self.Qn))
            if not self.symmetric:
                self.beta.set_value(min_a - self.s * self.Qn)
            self.init_state += 1
        elif self.init_state < self.batch_init:
            min_a = paddle.min(activation.detach())
            max_a = paddle.max(activation.detach())
            self.s.set_value(
                self.s * 0.9 + 0.1 * (max_a - min_a) / (self.Qp - self.Qn)
            )
            if not self.symmetric:
                self.beta.set_value(
                    self.s * 0.9 + 0.1 * (min_a - self.s * self.Qn)
                )
            self.init_state += 1
        else:
            self.init_state += 1
        activation.stop_gradient = False
        if not self.symmetric:
            q_a = LsqPlusActFunc.apply(
                activation, self.s, self.beta, self.g, self.Qn, self.Qp
            )
        else:
            q_a = LsqFunc.apply(
                activation, self.s, self.g, self.Qn, self.Qp, per_channel=False
            )
        return q_a


class FakeQuantWeightLSQPlus(Layer):
    def __init__(
        self,
        quant_bits,
        all_positive=False,
        per_channel=False,
        batch_init=20,
        channel_num=None,
        quant_linear=False,
        dtype='float32',
        name=None,
        reduce_type=None,
    ):
        super().__init__()
        '''
        Args:
            quant_bits(int): quantization bit number for weights.
            all_positive(bool): whether unsigned or signed quantization, where True for unsigned quantization and False for signed quantization.
            per_channel(bool): whether layer-wise or channel-wise quantization, where True for layer-wise quantization and False for channel-wise quantization.
            batch_init(int): number of batches that collect Gaussian approximation for the weight distribution in each layer.
            channel_num(int): the channel number of the weight which is needed when per_channel is True.
            quant_linear(bool): whether the weight is from Linear.
            dtype(str): data type.
            name(str): the name of the weight.
            reduce_type(str): the reduce type which is needed when parallel training.
        '''

        self.bits = quant_bits
        self.all_positive = all_positive
        self.per_channel = per_channel
        self.quant_linear = quant_linear
        self.batch_init = batch_init
        self.name = name
        self.quant_axis = 1 if quant_linear else 0
        self.collect_axis = 0 if quant_linear else 1
        self.reduce_type = reduce_type

        if self.all_positive:
            # unsigned weight
            self.Qn = 0
            self.Qp = 2**self.bits - 1
        else:
            # signed weight
            self.Qn = -(2 ** (self.bits - 1))
            self.Qp = 2 ** (self.bits - 1) - 1

        self.init_state = 0
        scale_prefix = f"{name}.scale" if name else 'quant_dequant.scale'
        self._scale_name = unique_name.generate(scale_prefix)
        s_attr = ParamAttr(
            name=self._scale_name, initializer=Constant(1.0), trainable=True
        )
        self.s = self.create_parameter(
            shape=[channel_num], attr=s_attr, dtype=dtype
        )
        self.s.stop_gradient = False

    def forward(self, weight):
        if self.reduce_type == "max":
            paddle.distributed.all_reduce(
                self.s, op=paddle.distributed.ReduceOp.MAX
            )

        if self.init_state == 0:
            self.g = paddle.to_tensor(1.0 / math.sqrt(weight.numel() * self.Qp))
            self.div = 2**self.bits - 1
            if self.per_channel:
                weight_tmp = weight.detach().reshape((weight.shape[0], -1))
                mean = paddle.mean(weight_tmp, axis=self.collect_axis)
                std = paddle.std(weight_tmp, axis=self.collect_axis)
                s = paddle.max(
                    paddle.stack(
                        [paddle.abs(mean - 3 * std), paddle.abs(mean + 3 * std)]
                    ),
                    axis=0,
                )
                self.s.set_value(s / self.div)
            else:
                mean = paddle.mean(weight.detach())
                std = paddle.std(weight.detach())
                self.s.set_value(
                    max(
                        [paddle.abs(mean - 3 * std), paddle.abs(mean + 3 * std)]
                    )
                    / self.div
                )
            self.init_state += 1
        elif self.init_state < self.batch_init:
            self.div = 2**self.bits - 1
            if self.per_channel:
                weight_tmp = weight.detach().reshape((weight.shape[0], -1))
                mean = paddle.mean(weight_tmp, axis=self.collect_axis)
                std = paddle.std(weight_tmp, axis=self.collect_axis)
                s = paddle.max(
                    paddle.stack(
                        [paddle.abs(mean - 3 * std), paddle.abs(mean + 3 * std)]
                    ),
                    axis=0,
                )
                self.s.set_value(s * 0.9 + 0.1 * s / self.div)
            else:
                mean = paddle.mean(weight.detach())
                std = paddle.std(weight.detach())
                self.s.set_value(
                    self.s * 0.9
                    + 0.1
                    * max(
                        [paddle.abs(mean - 3 * std), paddle.abs(mean + 3 * std)]
                    )
                    / self.div
                )
            self.init_state += 1
        elif self.init_state == self.batch_init:
            self.init_state += 1

        weight.stop_gradient = False
        w_q = LsqFunc.apply(
            weight,
            self.s,
            self.g,
            self.Qn,
            self.Qp,
            self.per_channel,
            self.quant_axis,
        )
        return w_q
