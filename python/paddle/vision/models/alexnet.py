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

import paddle
import paddle.nn as nn

__all__ = []


class AlexNet(nn.Layer):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        num_classes (int): output dim of last fc layer. Default: 1000.

    Examples:
        .. code-block:: python

            from paddle.vision.models import AlexNet

            model = AlexNet()
    """

    def __init__(self, num_classes: int=1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2D(
                3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=3, stride=2),
            nn.Conv2D(
                64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=3, stride=2),
            nn.Conv2D(
                192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(
                384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(
                256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=3, stride=2), )
        self.avgpool = nn.AdaptiveAvgPool2D((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes), )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, start_axis=1)
        x = self.classifier(x)
        return x
