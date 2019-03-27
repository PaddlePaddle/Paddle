# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from __future__ import print_function

import math

from .. import unique_name

__all__ = [
    'NoamDecay', 'PiecewiseDecay', 'NaturalExpDecay', 'ExponentialDecay',
    'InverseTimeDecay', 'CosineDecay'
]


class LearningRateDecay(object):
    """
    Base class of learning rate decay
    """

    def __init__(self, begin=0, step=1, dtype='float32'):
        self.step_num = begin
        self.step_size = step
        self.dtype = dtype

    def __call__(self):
        lr = self.step()
        if isinstance(lr, float):
            lr = self.create_lr_var(lr)
        self.step_num += self.step_size
        return lr

    def create_lr_var(self, lr):
        from .. import layers
        lr = layers.create_global_var(
            name=unique_name.generate("learning_rate"),
            shape=[1],
            value=float(lr),
            dtype=self.dtype,
            persistable=True)
        return lr

    def step(self):
        raise NotImplementedError()


class PiecewiseDecay(LearningRateDecay):
    def __init__(self, boundaries, values, begin, step=1, dtype='float32'):
        super(PiecewiseDecay, self).__init__(begin, step, dtype)
        self.boundaries = boundaries
        self.values = values

        self.vars = []
        for value in values:
            self.vars.append(self.create_lr_var(value))

    def step(self):
        for i in range(len(self.boundaries)):
            if self.step_num < self.boundaries[i]:
                return self.vars[i]
        return self.vars[len(self.values) - 1]


class NaturalExpDecay(LearningRateDecay):
    def __init__(self,
                 learning_rate,
                 decay_steps,
                 decay_rate,
                 staircase=False,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(NaturalExpDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def step(self):
        from .. import layers
        div_res = self.create_lr_var(self.step_num / self.decay_steps)
        if self.staircase:
            div_res = layers.floor(div_res)
        decayed_lr = self.learning_rate * layers.exp(-1 * self.decay_rate *
                                                     div_res)

        return decayed_lr


class ExponentialDecay(LearningRateDecay):
    def __init__(self,
                 learning_rate,
                 decay_steps,
                 decay_rate,
                 staircase=False,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(ExponentialDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def step(self):
        from .. import layers
        div_res = self.create_lr_var(self.step_num / self.decay_steps)
        if self.staircase:
            div_res = layers.floor(div_res)

        decayed_lr = self.learning_rate * (self.decay_rate**div_res)

        return decayed_lr


class InverseTimeDecay(LearningRateDecay):
    def __init__(self,
                 learning_rate,
                 decay_steps,
                 decay_rate,
                 staircase=False,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(InverseTimeDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def step(self):
        from .. import layers
        div_res = self.create_lr_var(self.step_num / self.decay_steps)
        if self.staircase:
            div_res = layers.floor(div_res)

        decayed_lr = self.learning_rate / (1 + self.decay_rate * div_res)

        return decayed_lr


class PolynomialDecay(LearningRateDecay):
    def __init__(self,
                 learning_rate,
                 decay_steps,
                 end_learning_rate=0.0001,
                 power=1.0,
                 cycle=False,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(PolynomialDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle

    def step(self):
        from .. import layers
        tmp_step_num = self.step_num
        tmp_decay_steps = self.decay_steps
        if self.cycle:
            div_res = layers.ceil(
                self.create_lr_var(tmp_step_num / self.decay_steps))
            zero_var = 0.0
            one_var = 1.0

            if float(tmp_step_num) == zero_var:
                div_res = one_var
            tmp_decay_steps = self.decay_steps * div_res
        else:
            tmp_step_num = self.create_lr_var(tmp_step_num
                                              if tmp_step_num < self.decay_steps
                                              else self.decay_steps)

        decayed_lr = (self.learning_rate - self.end_learning_rate) * \
            ((1 - tmp_step_num / tmp_decay_steps) ** self.power) + self.end_learning_rate
        return decayed_lr


class CosineDecay(LearningRateDecay):
    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(CosineDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.step_each_epoch = step_each_epoch
        self.epochs = epochs

    def step(self):
        from .. import layers
        cur_epoch = layers.floor(
            self.create_lr_var(self.step_num / self.step_each_epoch))
        decayed_lr = self.learning_rate * 0.5 * (
            layers.cos(cur_epoch * math.pi / self.epochs) + 1)
        return decayed_lr


class NoamDecay(LearningRateDecay):
    def __init__(self, d_model, warmup_steps, begin=1, step=1, dtype='float32'):
        super(NoamDecay, self).__init__(begin, step, dtype)
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def step(self):
        from .. import layers
        a = self.create_lr_var(self.step_num**-0.5)
        b = self.create_lr_var((self.warmup_steps**-1.5) * self.step_num)
        lr_value = (self.d_model**-0.5) * layers.elementwise_min(a, b)
        return lr_value
