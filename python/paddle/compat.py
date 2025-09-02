# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved
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
#
# This file implements most of the public API compatible with PyTorch.
# Note that this file does not depend on PyTorch in any way.
# This is a standalone implementation.

from .tensor.compat import (
    Unfold,
    max,
    median,
    min,
    nanmedian,
    sort,
    split,
)
from .tensor.compat_softmax import softmax

__all__ = [
    'softmax',
    'split',
    'sort',
    'Unfold',
    'min',
    'max',
    'median',
    'nanmedian',
]
