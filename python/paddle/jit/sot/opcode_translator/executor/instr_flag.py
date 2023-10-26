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

# flags for instructions


class FORMAT_VALUE_FLAG:
    FVC_MASK = 0x3
    FVC_NONE = 0x0
    FVC_STR = 0x1
    FVC_REPR = 0x2
    FVC_ASCII = 0x3
    FVS_MASK = 0x4
    FVS_HAVE_SPEC = 0x4


class MAKE_FUNCTION_FLAG:
    MF_HAS_CLOSURE = 0x08
    MF_HAS_ANNOTATION = 0x04
    MF_HAS_KWDEFAULTS = 0x02
    MF_HAS_DEFAULTS = 0x01


class CALL_FUNCTION_EX_FLAG:
    CFE_HAS_KWARGS = 0x01
