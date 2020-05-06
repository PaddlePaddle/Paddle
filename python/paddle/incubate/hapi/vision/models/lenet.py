#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, BatchNorm, Pool2D, Linear
from paddle.fluid.dygraph.container import Sequential

from ...model import Model

__all__ = ['LeNet']


class LeNet(Model):
    """LeNet model from
    `"LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11): 2278-2324.`_

    Args:
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer 
                            will not be defined. Default: 10.
        classifier_activation (str): activation for the last fc layer. Default: 'softmax'.

    Examples:
        .. code-block:: python

            from paddle.incubate.hapi.vision.models import LeNet

            model = LeNet()
    """

    def __init__(self, num_classes=10, classifier_activation='softmax'):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.features = Sequential(
            Conv2D(
                1, 6, 3, stride=1, padding=1),
            Pool2D(2, 'max', 2),
            Conv2D(
                6, 16, 5, stride=1, padding=0),
            Pool2D(2, 'max', 2))

        if num_classes > 0:
            self.fc = Sequential(
                Linear(400, 120),
                Linear(120, 84),
                Linear(
                    84, 10, act=classifier_activation))

    def forward(self, inputs):
        x = self.features(inputs)

        if self.num_classes > 0:
            x = fluid.layers.flatten(x, 1)
            x = self.fc(x)
        return x
