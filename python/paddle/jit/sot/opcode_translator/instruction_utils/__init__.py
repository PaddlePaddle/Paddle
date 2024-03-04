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

from .instruction_pass import apply_instr_pass  # noqa: F401
from .instruction_utils import (  # noqa: F401
    Instruction,
    Space,
    calc_offset_from_bytecode_offset,
    calc_stack_effect,
    convert_instruction,
    gen_instr,
    get_instructions,
    instrs_info,
    modify_extended_args,
    modify_instrs,
    modify_vars,
    relocate_jump_target,
    replace_instr,
    reset_offset,
)
from .opcode_analysis import (  # noqa: F401
    analysis_used_names,
)
