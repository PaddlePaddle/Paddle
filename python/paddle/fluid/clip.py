#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import warnings

import functools
import paddle
from . import layers
from . import framework
from . import core
from . import name_scope
from .dygraph import base as imperative_base
from .data_feeder import check_variable_and_dtype
from .framework import _non_static_mode, in_dygraph_mode, _in_legacy_dygraph
from .layer_helper import LayerHelper
from .framework import default_main_program
from paddle import _C_ops, _legacy_C_ops

__all__ = [
    'set_gradient_clip',
    'ClipGradByValue',
    'ClipGradByNorm',
    'ClipGradByGlobalNorm',
]
