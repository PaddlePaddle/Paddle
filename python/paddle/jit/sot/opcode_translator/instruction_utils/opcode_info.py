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
from __future__ import annotations

import opcode
import sys
from enum import Enum

REL_JUMP = {opcode.opname[x] for x in opcode.hasjrel}
REL_BWD_JUMP = {opname for opname in REL_JUMP if "BACKWARD" in opname}
REL_FWD_JUMP = REL_JUMP - REL_BWD_JUMP
ABS_JUMP = {opcode.opname[x] for x in opcode.hasjabs}
HAS_LOCAL = {opcode.opname[x] for x in opcode.haslocal}
HAS_FREE = {opcode.opname[x] for x in opcode.hasfree}
NEED_TO_BOOL = {"UNARY_NOT", "POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE"}
ALL_JUMP = REL_JUMP | ABS_JUMP
UNCONDITIONAL_JUMP = {"JUMP_ABSOLUTE", "JUMP_FORWARD"}
if sys.version_info >= (3, 11):
    UNCONDITIONAL_JUMP.add("JUMP_BACKWARD")
RETURN = {"RETURN_VALUE"}
if sys.version_info >= (3, 12):
    RETURN.add("RETURN_CONST")


class JumpDirection(Enum):
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"


class PopJumpCond(Enum):
    FALSE = "FALSE"
    TRUE = "TRUE"
    NONE = "NONE"
    NOT_NONE = "NOT_NONE"


def _get_pyopcode_cache_size() -> dict[str, int]:
    if sys.version_info >= (3, 11) and sys.version_info < (3, 12):
        # Cache for some opcodes, it's for Python 3.11
        # https://github.com/python/cpython/blob/3.11/Include/internal/pycore_opcode.h#L41-L53
        return {
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
    elif sys.version_info >= (3, 12) and sys.version_info < (3, 13):
        # Cache for some opcodes, it's for Python 3.12
        # https://github.com/python/cpython/blob/3.12/Include/internal/pycore_opcode.h#L34-L47
        return {
            "BINARY_SUBSCR": 1,
            "STORE_SUBSCR": 1,
            "UNPACK_SEQUENCE": 1,
            "FOR_ITER": 1,
            "STORE_ATTR": 4,
            "LOAD_ATTR": 9,
            "COMPARE_OP": 1,
            "LOAD_GLOBAL": 4,
            "BINARY_OP": 1,
            "SEND": 1,
            "LOAD_SUPER_ATTR": 1,
            "CALL": 3,
        }
    elif sys.version_info >= (3, 13) and sys.version_info < (3, 14):
        # Cache for some opcodes, it's for Python 3.13
        # https://github.com/python/cpython/blob/3.13/Include/internal/pycore_opcode_metadata.h#L1598-L1618
        return {
            "JUMP_BACKWARD": 1,
            "TO_BOOL": 3,
            "BINARY_SUBSCR": 1,
            "STORE_SUBSCR": 1,
            "SEND": 1,
            "UNPACK_SEQUENCE": 1,
            "STORE_ATTR": 4,
            "LOAD_GLOBAL": 4,
            "LOAD_SUPER_ATTR": 1,
            "LOAD_ATTR": 9,
            "COMPARE_OP": 1,
            "CONTAINS_OP": 1,
            "POP_JUMP_IF_TRUE": 1,
            "POP_JUMP_IF_FALSE": 1,
            "POP_JUMP_IF_NONE": 1,
            "POP_JUMP_IF_NOT_NONE": 1,
            "FOR_ITER": 1,
            "CALL": 3,
            "BINARY_OP": 1,
        }
    elif sys.version_info >= (3, 14):
        raise NotImplementedError(
            f"Need to supplement cache operation code, for Python {sys.version_info}"
        )
    else:
        return {}


PYOPCODE_CACHE_SIZE = _get_pyopcode_cache_size()
