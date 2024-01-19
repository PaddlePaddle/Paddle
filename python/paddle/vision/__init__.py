#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle  # noqa: F401
from paddle import nn  # noqa: F401

from . import (  # noqa: F401
    datasets,
    models,
    ops,
    transforms,
)
from .datasets import (  # noqa: F401
    MNIST,
    VOC2012,
    Cifar10,
    Cifar100,
    DatasetFolder,
    FashionMNIST,
    Flowers,
    ImageFolder,
)
from .image import (
    get_image_backend,
    image_load,
    set_image_backend,
)
from .models import (  # noqa: F401
    VGG,
    AlexNet,
    DenseNet,
    GoogLeNet,
    InceptionV3,
    LeNet,
    MobileNetV1,
    MobileNetV2,
    MobileNetV3Large,
    MobileNetV3Small,
    ResNet,
    ShuffleNetV2,
    SqueezeNet,
    alexnet,
    densenet121,
    densenet161,
    densenet169,
    densenet201,
    densenet264,
    googlenet,
    inception_v3,
    mobilenet_v1,
    mobilenet_v2,
    mobilenet_v3_large,
    mobilenet_v3_small,
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
    shufflenet_v2_swish,
    shufflenet_v2_x0_5,
    shufflenet_v2_x0_25,
    shufflenet_v2_x0_33,
    shufflenet_v2_x1_0,
    shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
    squeezenet1_0,
    squeezenet1_1,
    vgg11,
    vgg13,
    vgg16,
    vgg19,
    wide_resnet50_2,
    wide_resnet101_2,
)
from .transforms import (  # noqa: F401
    BaseTransform,
    BrightnessTransform,
    CenterCrop,
    ColorJitter,
    Compose,
    ContrastTransform,
    Grayscale,
    HueTransform,
    Normalize,
    Pad,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    SaturationTransform,
    ToTensor,
    Transpose,
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    center_crop,
    crop,
    hflip,
    normalize,
    pad,
    resize,
    rotate,
    to_grayscale,
    to_tensor,
    vflip,
)

__all__ = ['set_image_backend', 'get_image_backend', 'image_load']
