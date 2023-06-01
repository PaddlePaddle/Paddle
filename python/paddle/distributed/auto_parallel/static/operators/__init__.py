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

from .common import DistributedOperatorImplContainer
from .common import DistributedOperatorImpl
from .common import register_distributed_operator_impl_container
from .common import register_distributed_operator_impl
from .common import find_compatible_distributed_operator_impls
from . import dist_embedding
from . import dist_matmul
from . import dist_reshape
from . import dist_softmax
from . import dist_transpose
from . import dist_default
from . import dist_eltwise
from . import dist_check_finite_and_unscale
from . import dist_update_loss_scaling
from . import dist_split
from . import dist_fill_constant_batch_size_like
from . import dist_pnorm
from . import dist_slice
from . import dist_fused_feedforward
from . import dist_fused_attention
from . import dist_fused_dropout_add
from . import dist_reduce_sum_p
from . import dist_shape
from . import dist_assign
from . import dist_scale
from . import dist_dropout
from . import dist_flash_attn
