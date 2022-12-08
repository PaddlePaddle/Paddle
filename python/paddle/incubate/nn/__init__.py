#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from .layer.fused_transformer import FusedMultiHeadAttention  # noqa: F401
from .layer.fused_transformer import FusedFeedForward  # noqa: F401
from .layer.fused_transformer import FusedTransformerEncoderLayer  # noqa: F401
from .layer.fused_transformer import FusedMultiTransformer  # noqa: F401
from .layer.fused_linear import FusedLinear  # noqa: F401
from .layer.fused_transformer import (
    FusedBiasDropoutResidualLayerNorm,
)  # noqa: F401
from .layer.fused_ec_moe import FusedEcMoe  # noqa: F401

from .functional.fused_transformer import fused_feedforward  # noqa: F401
from .functional.fused_transformer import fused_bias_dropout_residual_layer_norm  # noqa: F401
from .functional.fused_transformer import fused_multi_head_attention  # noqa: F401
from .functional.fused_transformer import fused_multi_transformer  # noqa: F401
from .functional.fused_matmul_bias import fused_linear  # noqa: F401
from .functional.fused_matmul_bias import fused_matmul_bias  # noqa: F401
from .functional.fused_ec_moe import fused_ec_moe  # noqa: F401


__all__ = [  # noqa
    'FusedMultiHeadAttention',
    'FusedFeedForward',
    'FusedTransformerEncoderLayer',
    'FusedMultiTransformer',
    'FusedLinear',
    'FusedBiasDropoutResidualLayerNorm',
    'FusedEcMoe', 
    # functional
    'fused_feedforward', 
    'fused_bias_dropout_residual_layer_norm', 
    'fused_multi_head_attention', 
    'fused_multi_transformer', 
    'fused_linear', 
    'fused_matmul_bias', 
    'fused_ec_moe', 
]
