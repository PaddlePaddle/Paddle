#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from .clip_grad_norm_ import clip_grad_norm_
from .clip_grad_value_ import clip_grad_value_
from .spectral_norm_hook import spectral_norm
from .transform_parameters import (
    _stride_column,  # noqa: F401
    parameters_to_vector,
    vector_to_parameters,
)
from .weight_norm_hook import remove_weight_norm, weight_norm

__all__ = [
    'weight_norm',
    'remove_weight_norm',
    'spectral_norm',
    'parameters_to_vector',
    'vector_to_parameters',
    'clip_grad_norm_',
    'clip_grad_value_',
]
