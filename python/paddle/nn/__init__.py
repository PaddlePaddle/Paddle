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


from . import (  # noqa: F401
    attention,
    functional,
    init,
    initializer,
    quant,
    utils,
)
from .clip import ClipGradByGlobalNorm, ClipGradByNorm, ClipGradByValue
from .decode import BeamSearchDecoder, dynamic_decode

# TODO: remove loss, keep it for too many used in unittests
from .layer import loss  # noqa: F401
from .layer.activation import (
    CELU,
    ELU,
    GELU,
    GLU,
    SELU,
    Hardshrink,
    Hardsigmoid,
    Hardswish,
    Hardtanh,
    LeakyReLU,
    LogSigmoid,
    LogSoftmax,
    Maxout,
    Mish,
    PReLU,
    ReLU,
    ReLU6,
    RReLU,
    Sigmoid,
    Silu,
    Softmax,
    Softmax2D,
    Softplus,
    Softshrink,
    Softsign,
    Swish,
    Tanh,
    Tanhshrink,
    ThresholdedReLU,
)
from .layer.common import (
    AlphaDropout,
    Bilinear,
    CircularPad1D,
    CircularPad2D,
    CircularPad3D,
    ConstantPad1D,
    ConstantPad2D,
    ConstantPad3D,
    CosineSimilarity,
    Dropout,
    Dropout2D,
    Dropout3D,
    Embedding,
    FeatureAlphaDropout,
    Flatten,
    Fold,
    Identity,
    Linear,
    Pad1D,
    Pad2D,
    Pad3D,
    ReflectionPad1D,
    ReflectionPad2D,
    ReflectionPad3D,
    ReplicationPad1D,
    ReplicationPad2D,
    ReplicationPad3D,
    Unflatten,
    Unfold,
    Upsample,
    UpsamplingBilinear2D,
    UpsamplingNearest2D,
    ZeroPad1D,
    ZeroPad2D,
    ZeroPad3D,
)

# TODO: import all neural network related api under this directory,
# including layers, linear, conv, rnn etc.
from .layer.container import (
    LayerDict,
    LayerList,
    ParameterDict,
    ParameterList,
    Sequential,
)
from .layer.conv import (
    Conv1D,
    Conv1DTranspose,
    Conv2D,
    Conv2DTranspose,
    Conv3D,
    Conv3DTranspose,
)
from .layer.distance import PairwiseDistance
from .layer.layers import Layer
from .layer.loss import (
    AdaptiveLogSoftmaxWithLoss,
    BCELoss,
    BCEWithLogitsLoss,
    CosineEmbeddingLoss,
    CrossEntropyLoss,
    CTCLoss,
    GaussianNLLLoss,
    HingeEmbeddingLoss,
    HSigmoidLoss,
    KLDivLoss,
    L1Loss,
    MarginRankingLoss,
    MSELoss,
    MultiLabelMarginLoss,
    MultiLabelSoftMarginLoss,
    MultiMarginLoss,
    NLLLoss,
    PoissonNLLLoss,
    RNNTLoss,
    SmoothL1Loss,
    SoftMarginLoss,
    TripletMarginLoss,
    TripletMarginWithDistanceLoss,
)
from .layer.norm import (
    BatchNorm,
    BatchNorm1D,
    BatchNorm2D,
    BatchNorm3D,
    GroupNorm,
    InstanceNorm1D,
    InstanceNorm2D,
    InstanceNorm3D,
    LayerNorm,
    LocalResponseNorm,
    SpectralNorm,
    SyncBatchNorm,
)
from .layer.pooling import (
    AdaptiveAvgPool1D,
    AdaptiveAvgPool2D,
    AdaptiveAvgPool3D,
    AdaptiveMaxPool1D,
    AdaptiveMaxPool2D,
    AdaptiveMaxPool3D,
    AvgPool1D,
    AvgPool2D,
    AvgPool3D,
    FractionalMaxPool2D,
    FractionalMaxPool3D,
    LPPool1D,
    LPPool2D,
    MaxPool1D,
    MaxPool2D,
    MaxPool3D,
    MaxUnPool1D,
    MaxUnPool2D,
    MaxUnPool3D,
)
from .layer.rnn import (
    GRU,
    LSTM,
    RNN,
    BiRNN,
    GRUCell,
    LSTMCell,
    RNNCellBase,
    SimpleRNN,
    SimpleRNNCell,
)
from .layer.transformer import (
    MultiHeadAttention,
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from .layer.vision import ChannelShuffle, PixelShuffle, PixelUnshuffle
from .modules.container import (
    ModuleDict,
    ModuleList,
)
from .modules.module import Module
from .parameter import Parameter
from .utils.spectral_norm_hook import spectral_norm  # noqa: F401

SiLU = Silu
AdaptiveAvgPool1d = AdaptiveAvgPool1D
AdaptiveAvgPool2d = AdaptiveAvgPool2D
AdaptiveAvgPool3d = AdaptiveAvgPool3D
HuberLoss = SmoothL1Loss
MaxUnpool1d = MaxUnPool1D
MaxUnpool2d = MaxUnPool2D
MaxUnpool3d = MaxUnPool3D
UpsamplingBilinear2d = UpsamplingBilinear2D
UpsamplingNearest2d = UpsamplingNearest2D
ZeroPad1d = ZeroPad1D
ZeroPad2d = ZeroPad2D
ZeroPad3d = ZeroPad3D
ReflectionPad1d = ReflectionPad1D
ReflectionPad2d = ReflectionPad2D
ReflectionPad3d = ReflectionPad3D
ConstantPad1d = ConstantPad1D
ConstantPad2d = ConstantPad2D
ConstantPad3d = ConstantPad3D
ReplicationPad1d = ReplicationPad1D
ReplicationPad2d = ReplicationPad2D
ReplicationPad3d = ReplicationPad3D
CircularPad1d = CircularPad1D
CircularPad2d = CircularPad2D
CircularPad3d = CircularPad3D
Conv1d = Conv1D
Conv2d = Conv2D
Conv3d = Conv3D
AdaptiveMaxPool1d = AdaptiveMaxPool1D
AdaptiveMaxPool2d = AdaptiveMaxPool2D
AdaptiveMaxPool3d = AdaptiveMaxPool3D
LPPool2d = LPPool2D
LPPool1d = LPPool1D

__all__ = [
    'BatchNorm',
    'CELU',
    'GroupNorm',
    'LayerNorm',
    'SpectralNorm',
    'BatchNorm1D',
    'BatchNorm2D',
    'BatchNorm3D',
    'InstanceNorm1D',
    'InstanceNorm2D',
    'InstanceNorm3D',
    'SyncBatchNorm',
    'LocalResponseNorm',
    'Embedding',
    'Linear',
    'Upsample',
    'UpsamplingNearest2D',
    'UpsamplingBilinear2D',
    'Pad1D',
    'Pad2D',
    'Pad3D',
    'ConstantPad1D',
    'ConstantPad2D',
    'ConstantPad3D',
    'CircularPad1D',
    'CircularPad2D',
    'CircularPad3D',
    'ReplicationPad1D',
    'ReplicationPad2D',
    'ReplicationPad3D',
    'ReflectionPad1D',
    'ReflectionPad2D',
    'ReflectionPad3D',
    'CircularPad1d',
    'CircularPad2d',
    'CircularPad3d',
    'ConstantPad1d',
    'ConstantPad2d',
    'ConstantPad3d',
    'ReplicationPad1d',
    'ReplicationPad2d',
    'ReplicationPad3d',
    'ReflectionPad1d',
    'ReflectionPad2d',
    'ReflectionPad3d',
    'CosineSimilarity',
    'Dropout',
    'Dropout2D',
    'Dropout3D',
    'Bilinear',
    'AlphaDropout',
    'FeatureAlphaDropout',
    'Unfold',
    'Fold',
    'RNNCellBase',
    'SimpleRNNCell',
    'LSTMCell',
    'GRUCell',
    'RNN',
    'BiRNN',
    'SimpleRNN',
    'LSTM',
    'GRU',
    'dynamic_decode',
    'MultiHeadAttention',
    'Maxout',
    'Softsign',
    'Transformer',
    'MSELoss',
    'LogSigmoid',
    'BeamSearchDecoder',
    'ClipGradByNorm',
    'ReLU',
    'PairwiseDistance',
    'BCEWithLogitsLoss',
    'SmoothL1Loss',
    'MaxPool3D',
    'AdaptiveMaxPool2D',
    'Hardshrink',
    'Softplus',
    'KLDivLoss',
    'AvgPool2D',
    'L1Loss',
    'LeakyReLU',
    'AvgPool1D',
    'AdaptiveAvgPool3D',
    'AdaptiveMaxPool3D',
    'NLLLoss',
    'PoissonNLLLoss',
    'Conv1D',
    'Conv1d',
    'Sequential',
    'Hardswish',
    'Conv1DTranspose',
    'AdaptiveMaxPool1D',
    'TransformerEncoder',
    'Softmax',
    'Softmax2D',
    'ParameterDict',
    'ParameterList',
    'Conv2D',
    'Conv2d',
    'Softshrink',
    'Hardtanh',
    'TransformerDecoderLayer',
    'CrossEntropyLoss',
    'GELU',
    'GLU',
    'SELU',
    'Silu',
    'SiLU',
    'Conv2DTranspose',
    'CTCLoss',
    'RNNTLoss',
    'ThresholdedReLU',
    'AdaptiveAvgPool2D',
    'MaxPool1D',
    'Layer',
    'TransformerDecoder',
    'Conv3D',
    'Conv3d',
    'Tanh',
    'Conv3DTranspose',
    'Flatten',
    'AdaptiveAvgPool1D',
    'Tanhshrink',
    'HSigmoidLoss',
    'PReLU',
    'TransformerEncoderLayer',
    'AvgPool3D',
    'MaxPool2D',
    'MarginRankingLoss',
    'LayerList',
    'ClipGradByValue',
    'BCELoss',
    'Hardsigmoid',
    'ClipGradByGlobalNorm',
    'LogSoftmax',
    'Sigmoid',
    'Swish',
    'Mish',
    'PixelShuffle',
    'PixelUnshuffle',
    'ChannelShuffle',
    'ELU',
    'ReLU6',
    'LayerDict',
    'ZeroPad2D',
    'MaxUnPool1D',
    'MaxUnPool2D',
    'MaxUnPool3D',
    'MultiLabelSoftMarginLoss',
    'HingeEmbeddingLoss',
    'Identity',
    'CosineEmbeddingLoss',
    'RReLU',
    'MultiMarginLoss',
    'MultiLabelMarginLoss',
    'TripletMarginWithDistanceLoss',
    'TripletMarginLoss',
    'SoftMarginLoss',
    'GaussianNLLLoss',
    'AdaptiveLogSoftmaxWithLoss',
    'Unflatten',
    'FractionalMaxPool2D',
    'FractionalMaxPool3D',
    'LPPool1D',
    'LPPool2D',
    'ZeroPad1D',
    'ZeroPad3D',
    'Parameter',
    'AdaptiveMaxPool1d',
    'AdaptiveMaxPool2d',
    'AdaptiveMaxPool3d',
    'LPPool2d',
    'LPPool1d',
    'Module',
    'ModuleDict',
    'ModuleList',
]
