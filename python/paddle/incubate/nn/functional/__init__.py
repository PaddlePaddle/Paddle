# ruff: noqa: F401
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

from .blha_get_max_len import blha_get_max_len
from .block_multihead_attention import (
    block_multihead_attention,
    block_multihead_attention_xpu,
)

# from .moe_gate_dispatch_permute import moe_gate_dispatch_permute
from .build_src_rank_and_local_expert_id import (
    build_src_rank_and_local_expert_id,
)
from .cal_aux_loss import cal_aux_loss
from .expand_modality_expert_id import expand_modality_expert_id
from .fp8 import (
    fp8_gemm_blockwise,
    fp8_quant_blockwise,
    fused_act_dequant,
    fused_stack_transpose_quant,
    fused_swiglu_weighted_bwd,
    fused_transpose_split_quant,
    fused_transpose_wlch_split_quant,
    fused_weighted_swiglu_act_quant,
)
from .fused_bias_act import fused_bias_act
from .fused_dot_product_attention import (
    cudnn_flash_attention,
    fused_dot_product_attention,
)
from .fused_dropout_add import fused_dropout_add
from .fused_gate_attention import fused_gate_attention
from .fused_layer_norm import fused_layer_norm
from .fused_matmul_bias import (
    fused_linear,
    fused_linear_activation,
    fused_matmul_bias,
)
from .fused_rms_norm import fused_rms_norm
from .fused_rms_norm_ext import fused_rms_norm_ext
from .fused_rotary_position_embedding import fused_rotary_position_embedding
from .fused_transformer import (
    fused_bias_dropout_residual_layer_norm,
    fused_feedforward,
    fused_multi_head_attention,
    fused_multi_transformer,
)
from .int_bincount import int_bincount
from .masked_multihead_attention import masked_multihead_attention
from .moe_combine import moe_combine
from .moe_gate_dispatch import moe_gate_dispatch
from .moe_gate_dispatch_partial_nosoftmaxtopk import (
    moe_gate_dispatch_partial_nosoftmaxtopk,
)
from .moe_gate_dispatch_permute import moe_gate_dispatch_permute
from .swiglu import swiglu
from .variable_length_memory_efficient_attention import (
    variable_length_memory_efficient_attention,
)

__all__ = [
    'fp8_gemm_blockwise',
    'fp8_quant_blockwise',
    'fused_act_dequant',
    'fused_multi_head_attention',
    'fused_feedforward',
    'fused_multi_transformer',
    'fused_matmul_bias',
    'fused_linear',
    'fused_linear_activation',
    'fused_bias_dropout_residual_layer_norm',
    'fused_dropout_add',
    'fused_rotary_position_embedding',
    'fused_stack_transpose_quant',
    'fused_transpose_split_quant',
    'fused_transpose_wlch_split_quant',
    'variable_length_memory_efficient_attention',
    "fused_rms_norm",
    "fused_layer_norm",
    "fused_bias_act",
    'fused_swiglu_weighted_bwd',
    'fused_weighted_swiglu_act_quant',
    "masked_multihead_attention",
    "blha_get_max_len",
    "block_multihead_attention",
    "swiglu",
    "moe_combine",
    "expand_modality_expert_id",
    "cal_aux_loss",
    "build_src_rank_and_local_expert_id",
    "int_bincount",
    "fused_rms_norm_ext",
    "moe_gate_dispatch",
    "moe_gate_dispatch_permute",
    "moe_gate_dispatch_partial_nosoftmaxtopk",
]
