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

from .code_status import CodeStatus
from .exceptions import (
    BreakGraphError,
    FallbackError,
    InnerError,
    inner_error_default_handler,
)
from .magic_methods import magic_method_builtin_dispatch
from .paddle_api_config import (
    is_break_graph_tensor_methods,
    is_inplace_api,
    paddle_tensor_methods,
)
from .SotProfiler import EventGuard, SotProfiler, event_register
from .utils import (
    Cache,
    GraphLogger,
    NameGenerator,
    OrderedSet,
    ResumeFnNameFactory,
    Singleton,
    SotUndefinedVar,
    StepInfoManager,
    StepState,
    cost_model,
    count_if,
    current_tmp_name_records,
    execute_time,
    flatten_extend,
    get_unbound_method,
    hashable,
    in_paddle_module,
    is_break_graph_api,
    is_builtin_fn,
    is_clean_code,
    is_paddle_api,
    is_strict_mode,
    list_contain_by_id,
    list_find_index_by_id,
    log,
    log_do,
    map_if,
    map_if_extend,
    meta_str,
    min_graph_size,
    no_eval_frame,
    show_trackers,
    tmp_name_guard,
)

__all__ = [
    "InnerError",
    "FallbackError",
    "BreakGraphError",
    "Singleton",
    "NameGenerator",
    "OrderedSet",
    'inner_error_default_handler',
    "log",
    "log_do",
    "no_eval_frame",
    "is_builtin_fn",
    "is_paddle_api",
    "in_paddle_module",
    "is_break_graph_api",
    'is_break_graph_tensor_methods',
    "map_if",
    "map_if_extend",
    "flatten_extend",
    "count_if",
    "Cache",
    "execute_time",
    "magic_method_builtin_dispatch",
    "meta_str",
    "is_strict_mode",
    "is_clean_code",
    "paddle_tensor_methods",
    "ResumeFnNameFactory",
    "list_contain_by_id",
    "list_find_index_by_id",
    "show_trackers",
    "get_unbound_method",
    "GraphLogger",
    "SotUndefinedVar",
    "event_register",
    "EventGuard",
    "SotProfiler",
    "hashable",
    "is_inplace_api",
    "sotprof_range",
    "min_graph_size",
    "CodeStatus",
    "cost_model",
    "StepInfoManager",
    "StepState",
    "tmp_name_guard",
    "current_tmp_name_records",
]
