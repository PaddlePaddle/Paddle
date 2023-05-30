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

import opcode

UNARY = {
    "UNARY_POSITIVE",
    "UNARY_NEGATIVE",
    "UNARY_NOT",
    "UNARY_INVERT",
}

BINARY = {
    "BINARY_MATRIX_MULTIPLY",
    "BINARY_POWER",
    "BINARY_MULTIPLY",
    "BINARY_MODULO",
    "BINARY_ADD",
    "BINARY_SUBTRACT",
    "BINARY_SUBSCR",
    "BINARY_FLOOR_DIVIDE",
    "BINARY_TRUE_DIVIDE",
    "BINARY_LSHIFT",
    "BINARY_RSHIFT",
    "BINARY_AND",
    "BINARY_XOR",
    "BINARY_OR",
}

INPLACE = {
    "INPLACE_MATRIX_MULTIPLY",
    "INPLACE_FLOOR_DIVIDE",
    "INPLACE_TRUE_DIVIDE",
    "INPLACE_ADD",
    "INPLACE_SUBTRACT",
    "INPLACE_MULTIPLY",
    "INPLACE_MODULO",
    "INPLACE_POWER",
    "INPLACE_LSHIFT",
    "INPLACE_RSHIFT",
    "INPLACE_AND",
    "INPLACE_XOR",
    "INPLACE_OR",
}

CALL = {
    "CALL_FUNCTION",
    "CALL_FUNCTION_KW",
    "CALL_FUNCTION_EX",
    "CALL_METHOD",
}

COMPARE = {
    "COMPARE_OP",
}

IMPORT = {
    "IMPORT_FROM",
}

ITER = {
    "FOR_ITER",
}

LOAD = {
    "LOAD_BUILD_CLASS",
    "LOAD_CONST",
    "LOAD_NAME",
    "LOAD_ATTR",
    "LOAD_GLOBAL",
    "LOAD_FAST",
    "LOAD_CLOSURE",
    "LOAD_DEREF",
    "LOAD_CLASSDEREF",
    "LOAD_METHOD",
}

MAKE_FUNCTION = {
    "MAKE_FUNCTION",
}

UNPACK = {
    "UNPACK_SEQUENCE",
    "UNPACK_EX",
}


PUSH_ONE = (
    UNARY
    | BINARY
    | INPLACE
    | CALL
    | COMPARE
    | IMPORT
    | ITER
    | LOAD
    | MAKE_FUNCTION
)
PUSH_ARG = UNPACK

ALL_WITH_PUSH = PUSH_ONE | PUSH_ARG

REL_JUMP = {opcode.opname[x] for x in opcode.hasjrel}
ABS_JUMP = {opcode.opname[x] for x in opcode.hasjabs}
ALL_JUMP = REL_JUMP | ABS_JUMP

RETURN = {
    "RETURN_VALUE",
}
