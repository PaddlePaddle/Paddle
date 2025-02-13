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

from paddle.base.libpaddle.pir import (  # noqa: F401
    Block,
    CloneOptions,
    # drr
    DrrPatternContext,
    #
    IrMapping,
    Operation,
    OpOperand,
    PassManager,
    Program,
    Type,
    Value,
    check_unregistered_ops,
    create_shaped_type,
    fake_value,
    get_chunk_id,
    get_comp_op_name,
    get_current_insertion_point,
    get_op_role,
    is_fake_value,
    register_dist_dialect,
    register_paddle_dialect,
    reset_insertion_point_to_end,
    reset_insertion_point_to_start,
    set_chunk_id,
    set_comp_op_name,
    set_insertion_point,
    set_insertion_point_after,
    set_insertion_point_to_block_end,
    set_op_role,
    translate_to_pir,
    translate_to_pir_with_param_map,
    value_is_persistable,
)
from paddle.base.wrapped_decorator import signature_safe_contextmanager

from . import core  # noqa: F401
from .dtype_patch import monkey_patch_dtype  # noqa: F401
from .math_op_patch import monkey_patch_value  # noqa: F401
from .program_patch import monkey_patch_program  # noqa: F401


@signature_safe_contextmanager
def _optimized_guard(self, param_and_grads):
    try:
        yield
    finally:
        pass


Program._optimized_guard = _optimized_guard

__all__ = []
