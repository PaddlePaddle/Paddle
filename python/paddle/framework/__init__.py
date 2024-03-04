#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: import framework api under this directory

from ..base import core  # noqa: F401
from ..base.core import (  # noqa: F401
    CPUPlace,
    CUDAPinnedPlace,
    CUDAPlace,
    CustomPlace,
    IPUPlace,
    XPUPlace,
)
from ..base.dygraph import base  # noqa: F401
from ..base.dygraph.base import (  # noqa: F401
    disable_dygraph as enable_static,
    enable_dygraph as disable_static,
    grad,
    no_grad_ as no_grad,
)
from ..base.framework import (  # noqa: F401
    Block,
    IrGraph,
    OpProtoHolder,
    Parameter,
    Program,
    _apply_pass,
    _create_tensor,
    _current_expected_place,
    _current_expected_place_,
    _dygraph_tracer,
    _get_paddle_place,
    _global_flags,
    _set_expected_place,
    _stride_in_no_check_dy2st_diff as _no_check_dy2st_diff,
    convert_np_dtype_to_dtype_,
    deprecate_stat_dict,
    disable_signal_handler,
    dygraph_not_support,
    dygraph_only,
    generate_control_dev_var_name,
    get_flags,
    in_dygraph_mode as in_dynamic_mode,
    in_dynamic_or_pir_mode,
    in_pir_mode,
    set_flags,
    switch_main_program,
    switch_startup_program,
    use_pir_api,
)
from ..base.layer_helper import LayerHelper  # noqa: F401
from .io import async_save, clear_async_save_task_queue  # noqa: F401

# isort: off
# Do the *DUPLICATED* monkey-patch for the tensor object.
# We need remove the duplicated code here once we fix
# the illogical implement in the monkey-patch methods later.
from ..base.dygraph.math_op_patch import monkey_patch_math_tensor  # noqa: F401
from ..base.layers.math_op_patch import monkey_patch_variable  # noqa: F401

# isort: on
from ..base.param_attr import ParamAttr  # noqa: F401
from . import random  # noqa: F401
from .framework import get_default_dtype, set_default_dtype  # noqa: F401
from .io import load, save  # noqa: F401
from .io_utils import (  # noqa: F401
    _clone_var_in_block_,
    _load_program_scope,
    _open_file_buffer,
    _pack_loaded_dict,
    _pickle_loads_mac,
    _unpack_saved_dict,
    is_belong_to_optimizer,
    is_parameter,
    is_persistable,
)
from .random import seed  # noqa: F401

__all__ = []
