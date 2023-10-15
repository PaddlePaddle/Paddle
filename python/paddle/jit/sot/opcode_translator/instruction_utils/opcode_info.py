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

import sys
from enum import Enum

import opcode

REL_JUMP = {opcode.opname[x] for x in opcode.hasjrel}
REL_BWD_JUMP = {opname for opname in REL_JUMP if "BACKWARD" in opname}
REL_FWD_JUMP = REL_JUMP - REL_BWD_JUMP
ABS_JUMP = {opcode.opname[x] for x in opcode.hasjabs}
HAS_LOCAL = {opcode.opname[x] for x in opcode.haslocal}
HAS_FREE = {opcode.opname[x] for x in opcode.hasfree}
ALL_JUMP = REL_JUMP | ABS_JUMP
UNCONDITIONAL_JUMP = {"JUMP_ABSOLUTE", "JUMP_FORWARD"}
if sys.version_info >= (3, 11):
    UNCONDITIONAL_JUMP.add("JUMP_BACKWARD")


class JumpDirection(Enum):
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"


class PopJumpCond(Enum):
    FALSE = "FALSE"
    TRUE = "TRUE"
    NONE = "NONE"
    NOT_NONE = "NOT_NONE"


# Cache for some opcodes, it's for Python 3.11+
# https://github.com/python/cpython/blob/3.11/Include/internal/pycore_opcode.h#L41-L53
PYOPCODE_CACHE_SIZE = {
    "BINARY_SUBSCR": 4,
    "STORE_SUBSCR": 1,
    "UNPACK_SEQUENCE": 1,
    "STORE_ATTR": 4,
    "LOAD_ATTR": 4,
    "COMPARE_OP": 2,
    "LOAD_GLOBAL": 5,
    "BINARY_OP": 1,
    "LOAD_METHOD": 10,
    "PRECALL": 1,
    "CALL": 4,
}
