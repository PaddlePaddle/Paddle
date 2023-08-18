# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

from .fused_transformer import fused_multi_head_attention
from .fused_transformer import fused_feedforward
from .fused_transformer import fused_multi_transformer
from .fused_matmul_bias import (
    fused_matmul_bias,
    fused_linear,
    fused_linear_activation,
)
from .fused_transformer import fused_bias_dropout_residual_layer_norm
from .fused_ec_moe import fused_ec_moe
from .fused_dropout_add import fused_dropout_add
from .fused_gate_attention import fused_gate_attention
from .fused_rotary_position_embedding import fused_rotary_position_embedding
from .variable_length_memory_efficient_attention import (
    variable_length_memory_efficient_attention,
)
from .fused_rms_norm import fused_rms_norm
from .fused_layer_norm import fused_layer_norm
from .masked_multihead_attention import masked_multihead_attention
from .masked_multiquery_attention import masked_multiquery_attention

__all__ = [
    'fused_multi_head_attention',
    'fused_feedforward',
    'fused_multi_transformer',
    'fused_matmul_bias',
    'fused_linear',
    'fused_linear_activation',
    'fused_bias_dropout_residual_layer_norm',
    'fused_ec_moe',
    'fused_dropout_add',
    'fused_rotary_position_embedding',
    'variable_length_memory_efficient_attention',
    "fused_rms_norm",
    "fused_layer_norm",
    "masked_multihead_attention",
    "masked_multiquery_attention",
]
