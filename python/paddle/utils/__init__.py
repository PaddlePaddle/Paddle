# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from ..base.framework import require_version
from . import (  # noqa: F401
    cpp_extension,
    dlpack,
    download,
    image_util,
    layers_utils,
    unique_name,
)
from .deprecated import deprecated
from .install_check import run_check
from .layers_utils import (  # noqa: F401
    _contain_var,
    _convert_to_tensor_list,
    _hash_with_id,
    _is_symmetric_padding,
    _packed_nest_with_indices,
    _recursive_assert_same_structure,
    _sequence_like,
    _yield_flat_nest,
    _yield_value,
    assert_same_structure,
    check_shape,
    convert_shape_to_list,
    convert_to_list,
    copy_mutable_vars,
    flatten,
    get_inputs_outputs_in_block,
    get_int_tensor_list,
    get_shape_tensor_inputs,
    hold_mutable_vars,
    is_sequence,
    map_structure,
    pack_sequence_as,
    padding_to_same_structure,
    to_sequence,
    try_get_constant_shape_from_tensor,
    try_set_static_shape_tensor,
)
from .lazy_import import try_import
from .op_version import OpLastCheckpointChecker  # noqa: F401

__all__ = ['deprecated', 'run_check', 'require_version', 'try_import']
