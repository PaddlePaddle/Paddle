# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
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

from .asp import (
    ASPHelper,  # noqa: F401
    decorate,
    prune_model,
    reset_excluded_layers,
    set_excluded_layers,
)
from .supported_layer_list import add_supported_layer
from .utils import (  # noqa: F401
    CheckMethod,
    MaskAlgo,
    calculate_density,
    check_mask_1d,
    check_mask_2d,
    check_sparsity,
    create_mask,
    get_mask_1d,
    get_mask_2d_best,
    get_mask_2d_greedy,
)

__all__ = [
    'calculate_density',
    'decorate',
    'prune_model',
    'set_excluded_layers',
    'reset_excluded_layers',
    'add_supported_layer',
]
