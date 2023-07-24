# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from ..wrapped_decorator import signature_safe_contextmanager

from .layer_function_generator import templatedoc
from .. import core
from ..framework import (
    Program,
    Variable,
    Operator,
    static_only,
    in_dygraph_mode,
)
from ..layer_helper import LayerHelper, unique_name
from ...utils import (
    assert_same_structure,
    map_structure,
    hold_mutable_vars,
    copy_mutable_vars,
    is_sequence,
    pack_sequence_as,
    flatten,
    to_sequence,
)
import numpy
import warnings
from functools import reduce, partial
from ..data_feeder import (
    convert_dtype,
    check_variable_and_dtype,
    check_type,
    check_dtype,
)
from ..backward import _infer_var_data_type_shape_
import paddle
from paddle import _C_ops, _legacy_C_ops

__all__ = []
