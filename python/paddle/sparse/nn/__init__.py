#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from . import functional

from .layer.activation import ReLU
from .layer.norm import BatchNorm, SyncBatchNorm
from .layer.activation import Softmax
from .layer.activation import ReLU6
from .layer.activation import LeakyReLU
from .layer.conv import Conv3D
from .layer.conv import Conv2D
from .layer.conv import SubmConv2D
from .layer.conv import SubmConv3D
from .layer.pooling import MaxPool3D

__all__ = [
    'ReLU',
    'ReLU6',
    'LeakyReLU',
    'Softmax',
    'BatchNorm',
    'SyncBatchNorm',
    'Conv2D',
    'Conv3D',
    'SubmConv2D',
    'SubmConv3D',
    'MaxPool3D',
]
