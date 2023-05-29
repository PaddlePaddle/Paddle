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

from .optimizer import LookAhead  # noqa: F401
from .optimizer import ModelAverage  # noqa: F401
from .optimizer import DistributedFusedLamb  # noqa: F401
from .checkpoint import auto_checkpoint  # noqa: F401
from ..fluid.layer_helper import LayerHelper  # noqa: F401
from .operators import softmax_mask_fuse_upper_triangle  # noqa: F401
from .operators import softmax_mask_fuse  # noqa: F401
from .operators import graph_send_recv
from .operators import graph_khop_sampler
from .operators import graph_sample_neighbors
from .operators import graph_reindex
from .tensor import segment_sum
from .tensor import segment_mean
from .tensor import segment_max
from .tensor import segment_min
from .tensor import _npu_identity
from .passes import fuse_resnet_unit_pass

from . import autograd  # noqa: F401
from . import autotune  # noqa: F401
from . import nn  # noqa: F401
from . import asp  # noqa: F401
from . import multiprocessing  # noqa: F401
from . import layers

from .nn.loss import identity_loss

from ..distributed import fleet
from . import xpu

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
]
