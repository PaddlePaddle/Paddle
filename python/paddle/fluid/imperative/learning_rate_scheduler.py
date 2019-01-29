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

from .. import layers
from .. import unique_name

__all__ = [
    'ExponentialDecay', 'NaturalExpDecay', 'InverseTimeDecay',
    'PolynomialDecay', 'PiecewiseDecay', 'NoamDecay'
]


class LearningRateDecay(object):
    """
    Base class of learning rate decay
    """

    def __init__(self, step, dtype='float32'):
        self.step = step
        self.dtype = dtype

    def __call__(self):
        lr = self.step()
        if isinstance(lr, float):
            lr = self._create_lr_var(lr)
        self.step += 1
        return lr

    def create_lr_var(lr):
        lr = layers.create_global_var(
            name=unique_name.generate("learning_rate"),
            shape=[1],
            value=float(lr),
            dtype=self.dtype,
            persistable=True)

    def step(self):
        raise NotImplementedError()


class PiecewiseDecay(object):
    def __init__(self, boundaries, values, step, dtype='float32'):
        super(PiecewiseDecay, self).__init__(step, dtype)
        self.boundaries = boundaries
        self.values = values

        self.vars = []
        for value in values:
            self.vars.append(self.create_lr_var(value))

    def step(self):
        for i in range(len(boundaries)):
            if self.step <= boundaries[i]:
                return self.vars[i]
        return self.vars[len(values) - 1]
