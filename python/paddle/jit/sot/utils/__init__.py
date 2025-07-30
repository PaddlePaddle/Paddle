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
    ENV_MIN_GRAPH_SIZE,
    ENV_SOT_ALLOW_DYNAMIC_SHAPE,
    ENV_SOT_CE_DEBUG_MODE,
    ENV_SOT_COLLECT_INFO,
    ENV_SOT_ENABLE_0_SIZE_FALLBACK,
    ENV_SOT_ENABLE_FASTER_GUARD,
    ENV_SOT_ENABLE_GUARD_TREE,
    ENV_SOT_ENABLE_STRICT_GUARD_CHECK,
    ENV_SOT_EXPORT,
    ENV_SOT_FORCE_FALLBACK_SIR_IDS,
    ENV_SOT_LOG_LEVEL,
    ENV_SOT_SERIALIZE_INFO,
    ENV_SOT_TRACE_NUMPY,
    ENV_SOT_UNSAFE_CACHE_FASTPATH,
    ENV_SOT_WITH_CONTROL_FLOW,
    ENV_STRICT_MODE,
    PEP508LikeEnvironmentVariable,
    allow_dynamic_shape_guard,
    enable_0_size_fallback_guard,
    export_guard,
    faster_guard_guard,
    guard_tree_guard,
    min_graph_size_guard,
    sot_step_profiler_guard,
    specialized_dim_numbers_guard,
    strict_mode_guard,
    with_control_flow_guard,
)
from .exceptions import (  # noqa: F401
    BreakGraphError,
    BreakGraphReasonBase,
    BuiltinFunctionBreak,
    ConditionalFallbackError,
    DataDependencyControlFlowBreak,
    DataDependencyDynamicShapeBreak,
    DataDependencyOperationBreak,
    ExportError,
    FallbackError,
    InnerError,
    PsdbBreakReason,
    SotCapturedException,
    SotCapturedExceptionFactory,
    SotErrorBase,
    UnsupportedIteratorBreak,
    UnsupportedOperationBreak,
    inner_error_default_handler,
)
from .info_collector import (  # noqa: F401
    BreakGraphReasonInfo,
    CompileCountInfo,
    InfoCollector,
    NewSymbolHitRateInfo,
    SubGraphInfo,
    SubGraphRelationInfo,
)
from .magic_methods import magic_method_builtin_dispatch  # noqa: F401
from .numpy_utils import (  # noqa: F401
    NUMPY_API_SUPPORTED_DICT,
)
from .paddle_api_config import (  # noqa: F401
    get_tensor_methods,
    is_break_graph_tensor_methods,
    is_directly_run_api,
    is_inplace_api,
    is_not_supported_paddle_layer,
)
from .utils import (  # noqa: F401
    Cache,
    ConstTypes,
    NameGenerator,
    ResumeFnNameFactory,
    Singleton,
    SIRToCodeMap,
    SotUndefinedVar,
    StepInfoManager,
    already_unified_in_dynamic_and_static_graph,
    count_if,
    current_symbol_registry,
    do_until_stop_iteration,
    execute_time,
    flatten,
    flatten_extend,
    get_api_fullname,
    get_min_non_specialized_number,
    get_numpy_ufuncs,
    get_obj_stable_repr,
    get_unbound_method,
    hashable,
    in_paddle_module,
    is_break_graph_api,
    is_builtin_fn,
    is_comprehensive_name,
    is_namedtuple_class,
    is_paddle_api,
    is_strict_mode,
    list_contain_by_id,
    list_find_index_by_id,
    log,
    log_do,
    log_enabled,
    log_format,
    log_once,
    map_if,
    map_if_extend,
    meta_str,
    no_eval_frame,
    printable,
    switch_symbol_registry,
    update_list_inplace,
)
