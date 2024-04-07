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

# TODO: define activation functions of neural network

from . import container, rnn, transformer  # noqa: F401
from .activation import (  # noqa: F401
    CELU,
    LeakyReLU,
    LogSoftmax,
    PReLU,
    ReLU,
    ReLU6,
    RReLU,
    Sigmoid,
    Softmax,
    Softmax2D,
)
from .common import (  # noqa: F401
    AlphaDropout,
    Bilinear,
    CosineSimilarity,
    Dropout,
    Dropout2D,
    Dropout3D,
    Embedding,
    Flatten,
    Fold,
    Identity,
    Linear,
    Pad1D,
    Pad2D,
    Pad3D,
    Unflatten,
    Upsample,
    UpsamplingBilinear2D,
    UpsamplingNearest2D,
    ZeroPad2D,
)
from .container import LayerDict  # noqa: F401
from .conv import (  # noqa: F401
    Conv1D,
    Conv1DTranspose,
    Conv2D,
    Conv2DTranspose,
    Conv3D,
    Conv3DTranspose,
)
from .distance import PairwiseDistance  # noqa: F401
from .layers import Layer  # noqa: F401
from .loss import (  # noqa: F401
    AdaptiveLogSoftmaxWithLoss,
    BCELoss,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    CTCLoss,
    GaussianNLLLoss,
    HingeEmbeddingLoss,
    KLDivLoss,
    L1Loss,
    MarginRankingLoss,
    MSELoss,
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
from .norm import (  # noqa: F401
    BatchNorm1D,
    BatchNorm2D,
    BatchNorm3D,
    GroupNorm,
    LayerNorm,
    LocalResponseNorm,
    SpectralNorm,
    SyncBatchNorm,
)
from .pooling import (  # noqa: F401
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
    MaxPool1D,
    MaxPool2D,
    MaxPool3D,
    MaxUnPool1D,
    MaxUnPool2D,
    MaxUnPool3D,
)
from .vision import ChannelShuffle, PixelShuffle, PixelUnshuffle  # noqa: F401

__all__ = []
