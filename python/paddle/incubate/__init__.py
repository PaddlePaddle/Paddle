# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

from ..base.layer_helper import LayerHelper  # noqa: F401
from ..distributed import fleet  # noqa: F401
from . import (  # noqa: F401
    asp,
    autograd,
    autotune,
    layers,
    multiprocessing,
    nn,
    xpu,
)
from .checkpoint import auto_checkpoint  # noqa: F401
from .framework import (  # noqa: F401
    get_rng_state,
    register_rng_state_as_index,
    set_rng_state,
)
from .jit import inference
from .nn.loss import identity_loss
from .operators import (
    graph_khop_sampler,
    graph_reindex,
    graph_sample_neighbors,
    graph_send_recv,
    softmax_mask_fuse,
    softmax_mask_fuse_upper_triangle,
)
from .optimizer import (
    DistributedFusedLamb,  # noqa: F401
    LookAhead,
    ModelAverage,
)
from .passes import fuse_resnet_unit_pass  # noqa: F401
from .tensor import (
    _npu_identity,  # noqa: F401
    segment_max,
    segment_mean,
    segment_min,
    segment_sum,
)

__all__ = [
    'LookAhead',
    'ModelAverage',
    'softmax_mask_fuse_upper_triangle',
    'softmax_mask_fuse',
    'graph_send_recv',
    'graph_khop_sampler',
    'graph_sample_neighbors',
    'graph_reindex',
    'segment_sum',
    'segment_mean',
    'segment_max',
    'segment_min',
    'identity_loss',
    'inference',
]
