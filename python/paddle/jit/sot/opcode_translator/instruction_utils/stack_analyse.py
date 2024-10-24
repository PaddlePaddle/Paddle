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

from ...utils import Singleton


class StackAnalyser(metaclass=Singleton):
    def stack_effect(self, instr):
        if "BINARY" in instr.opname or "INPLACE" in instr.opname:
            return 2, 1
        elif "UNARY" in instr.opname:
            return 1, 1
        return getattr(self, instr.opname)(instr)

    def LOAD_GLOBAL(self, instr):
        return 0, 1

    def LOAD_CONST(self, instr):
        return 0, 1

    def LOAD_FAST(self, instr):
        return 0, 1

    def LOAD_ATTR(self, instr):
        return 1, 1

    def LOAD_METHOD(self, instr):
        return 1, 2

    def STORE_FAST(self, instr):
        return 1, 0

    def BUILD_TUPLE(self, instr):
        return instr.arg, 1

    def BUILD_LIST(self, instr):
        return instr.arg, 1

    def BUILD_SLICE(self, instr):
        if instr.arg == 3:
            return 3, 1
        else:
            return 2, 1

    def UNPACK_SEQUENCE(self, instr):
        return 1, instr.arg

    def CALL_FUNCTION(self, instr):
        return instr.arg + 1, 1

    def DUP_TOP(self, instr):
        return 0, 1

    def DUP_TOP_TWO(self, instr):
        return 0, 2

    def ROT_N(self, instr):
        return 0, 0

    def ROT_TWO(self, instr):
        return 0, 0

    def ROT_THREE(self, instr):
        return 0, 0

    def ROT_FOUR(self, instr):
        return 0, 0

    def GET_ITER(self, instr):
        return 1, 1

    def POP_TOP(self, instr):
        return 1, 0

    def PUSH_NULL(self, instr):
        return 0, 1

    def NOP(self, instr):
        return 0, 0

    def EXTENDED_ARG(self, instr):
        return 0, 0
