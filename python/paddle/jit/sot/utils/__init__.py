# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from .call_ast_utils import get_static_function, try_ast_func  # noqa: F401
from .envs import (  # noqa: F401
    ENV_CLEAN_CODE,
    ENV_COST_MODEL,
    ENV_MIN_GRAPH_SIZE,
    ENV_SOT_ALLOW_DYNAMIC_SHAPE,
    ENV_SOT_EXPORT,
    ENV_SOT_LOG_LEVEL,
    ENV_SOT_WITH_CONTROL_FLOW,
    ENV_STRICT_MODE,
    cost_model_guard,
    min_graph_size_guard,
    strict_mode_guard,
    with_allow_dynamic_shape_guard,
    with_control_flow_guard,
    with_export_guard,
)
from .exceptions import (  # noqa: F401
    BreakGraphError,
    ExportError,
    FallbackError,
    InnerError,
    inner_error_default_handler,
)
from .magic_methods import magic_method_builtin_dispatch  # noqa: F401
from .paddle_api_config import (  # noqa: F401
    is_break_graph_tensor_methods,
    is_inplace_api,
    is_not_supported_paddle_layer,
    paddle_tensor_methods,
)
from .utils import (  # noqa: F401
    Cache,
    ConstTypes,
    GraphLogger,
    NameGenerator,
    ResumeFnNameFactory,
    Singleton,
    SotUndefinedVar,
    StepInfoManager,
    StepState,
    count_if,
    current_tmp_name_records,
    execute_time,
    flatten,
    flatten_extend,
    get_api_fullname,
    get_unbound_method,
    hashable,
    in_paddle_module,
    is_break_graph_api,
    is_builtin_fn,
    is_clean_code,
    is_comprehensive_name,
    is_paddle_api,
    is_strict_mode,
    list_contain_by_id,
    list_find_index_by_id,
    log,
    log_do,
    log_format,
    map_if,
    map_if_extend,
    meta_str,
    no_eval_frame,
    printable,
    tmp_name_guard,
)
