# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License
import os

from . import (  # noqa: F401
    dist_assign,
    dist_check_finite_and_unscale,
    dist_concat,
    dist_default,
    dist_dropout,
    dist_eltwise,
    dist_embedding,
    dist_expand_as,
    dist_fill_constant_batch_size_like,
    dist_flash_attn,
    dist_fused_attention,
    dist_fused_dropout_add,
    dist_fused_feedforward,
    dist_fused_rms_norm,
    dist_fused_rope,
    dist_gather_nd,
    dist_layer_norm,
    dist_matmul,
    dist_pnorm,
    dist_reduce_sum_p,
    dist_reshape,
    dist_scale,
    dist_shape,
    dist_slice,
    dist_softmax,
    dist_split,
    dist_stack,
    dist_strided_slice,
    dist_tile,
    dist_transpose,
    dist_unsqueeze2,
    dist_update_loss_scaling,
)
from .common import (  # noqa: F401
    DistributedOperatorImpl,
    DistributedOperatorImplContainer,
    find_compatible_distributed_operator_impls,
    find_distributed_operator_impl_container,
    register_distributed_operator_impl,
    register_distributed_operator_impl_container,
)

parallel_ce = os.getenv("PARALLEL_CROSS_ENTROPY")
if parallel_ce == "true":
    from . import dist_cross_entropy  # noqa: F401
