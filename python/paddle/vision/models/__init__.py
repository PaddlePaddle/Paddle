#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .alexnet import AlexNet, alexnet
from .densenet import (
    DenseNet,
    densenet121,
    densenet161,
    densenet169,
    densenet201,
    densenet264,
)
from .googlenet import GoogLeNet, googlenet
from .inceptionv3 import InceptionV3, inception_v3
from .lenet import LeNet
from .mobilenetv1 import MobileNetV1, mobilenet_v1
from .mobilenetv2 import MobileNetV2, mobilenet_v2
from .mobilenetv3 import (
    MobileNetV3Large,
    MobileNetV3Small,
    mobilenet_v3_large,
    mobilenet_v3_small,
)
from .resnet import (
    ResNet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext50_64x4d,
    resnext101_32x4d,
    resnext101_64x4d,
    resnext152_32x4d,
    resnext152_64x4d,
    wide_resnet50_2,
    wide_resnet101_2,
)
from .shufflenetv2 import (
    ShuffleNetV2,
    shufflenet_v2_swish,
    shufflenet_v2_x0_5,
    shufflenet_v2_x0_25,
    shufflenet_v2_x0_33,
    shufflenet_v2_x1_0,
    shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
)
from .squeezenet import SqueezeNet, squeezenet1_0, squeezenet1_1
from .vgg import VGG, vgg11, vgg13, vgg16, vgg19

__all__ = [
    'ResNet',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnext50_32x4d',
    'resnext50_64x4d',
    'resnext101_32x4d',
    'resnext101_64x4d',
    'resnext152_32x4d',
    'resnext152_64x4d',
    'wide_resnet50_2',
    'wide_resnet101_2',
    'VGG',
    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19',
    'MobileNetV1',
    'mobilenet_v1',
    'MobileNetV2',
    'mobilenet_v2',
    'MobileNetV3Small',
    'MobileNetV3Large',
    'mobilenet_v3_small',
    'mobilenet_v3_large',
    'LeNet',
    'DenseNet',
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
    'densenet264',
    'AlexNet',
    'alexnet',
    'InceptionV3',
    'inception_v3',
    'SqueezeNet',
    'squeezenet1_0',
    'squeezenet1_1',
    'GoogLeNet',
    'googlenet',
    'ShuffleNetV2',
    'shufflenet_v2_x0_25',
    'shufflenet_v2_x0_33',
    'shufflenet_v2_x0_5',
    'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5',
    'shufflenet_v2_x2_0',
    'shufflenet_v2_swish',
]
