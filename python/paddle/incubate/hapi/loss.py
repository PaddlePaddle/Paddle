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

import os

from paddle import fluid
from paddle.fluid.framework import in_dygraph_mode, Variable
from paddle.fluid.dygraph.base import to_variable

from .utils import to_list

__all__ = ['Loss', 'CrossEntropy', 'SoftmaxWithCrossEntropy']


class Loss(object):
    """
    Base class for loss, encapsulates loss logic and APIs

    Usage:
        custom_loss = CustomLoss()
        loss = custom_loss(inputs, labels)
    
    Examples:
        .. code-block:: python

            from paddle.incubate.hapi.loss import Loss
            from paddle import fluid

            class SoftmaxWithCrossEntropy(Loss):
                def __init__(self, average=True):
                    super(SoftmaxWithCrossEntropy, self).__init__(average)

                def forward(self, outputs, labels):
                    return [
                        fluid.layers.softmax_with_cross_entropy(
                            o, l, return_softmax=False) for o, l in zip(outputs, labels)
                    ]
            
    """

    def __init__(self, average=True):
        super(Loss, self).__init__()
        self.average = average

    def forward(self, outputs, labels):
        raise NotImplementedError()

    def __call__(self, outputs, labels=None):
        labels = to_list(labels)
        if in_dygraph_mode() and labels:
            labels = [to_variable(l) for l in labels]
        losses = to_list(self.forward(to_list(outputs), labels))
        if self.average:
            losses = [fluid.layers.reduce_mean(l) for l in losses]
        else:
            losses = [fluid.layers.reduce_sum(l) for l in losses]
        return losses


class CrossEntropy(Loss):
    """
    Args:
        input (list[Variable]): Input tensor, the data type is float32,
            float64, int32, int64.
        label (list[Variable]): Label tensor, the data type is float32,
            float64, int32, int64.
        average (bool, optional): Indicate whether to average the loss, Default: True.
    Returns:
        list[Variable]: The tensor variable storing the cross_entropy_loss of inputs and labels.

    Examples:
        .. code-block:: python

            from paddle.incubate.hapi.model import Input
            from paddle.incubate.hapi.vision.models import LeNet
            from paddle.incubate.hapi.loss import CrossEntropy

            inputs = [Input([-1, 1, 28, 28], 'float32', name='image')]
            labels = [Input([None, 1], 'int64', name='label')]

            model = LeNet()
            loss = CrossEntropy()
            model.prepare(loss_function=loss, inputs=inputs, labels=labels)
            
    """

    def __init__(self, average=True):
        super(CrossEntropy, self).__init__(average)

    def forward(self, outputs, labels):
        return [
            fluid.layers.cross_entropy(o, l) for o, l in zip(outputs, labels)
        ]


class SoftmaxWithCrossEntropy(Loss):
    """
    this op combined softmax and cross entropy.
    Args:
        input (list[Variable]): Input tensor, the data type is float32,
            float64, int32, int64.
        label (list[Variable]): Label tensor, the data type is float32,
            float64, int32, int64.
        average (bool, optional): Indicate whether to average the loss, Default: True.
    Returns:
        list[Variable]: The tensor variable storing the cross_entropy_loss of inputs and labels.

    Examples:
        .. code-block:: python

            from paddle.incubate.hapi.model import Input
            from paddle.incubate.hapi.vision.models import LeNet
            from paddle.incubate.hapi.loss import SoftmaxWithCrossEntropy

            inputs = [Input([-1, 1, 28, 28], 'float32', name='image')]
            labels = [Input([None, 1], 'int64', name='label')]

            model = LeNet(classifier_activation=None)
            loss = SoftmaxWithCrossEntropy()
            model.prepare(loss_function=loss, inputs=inputs, labels=labels)
    """

    def __init__(self, average=True):
        super(SoftmaxWithCrossEntropy, self).__init__(average)

    def forward(self, outputs, labels):
        return [
            fluid.layers.softmax_with_cross_entropy(
                o, l, return_softmax=False) for o, l in zip(outputs, labels)
        ]
