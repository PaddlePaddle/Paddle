# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


def get_dtype_lower_case_name(dtype):
    return _up_case_name2lower_case_name[dtype.name]


_up_case_name2lower_case_name = {
    "UNDEFINED": "void",
    "BOOL": "bool",
    "INT8": "int8",
    "UINT8": "uint8",
    "INT16": "int16",
    "UINT16": "uint16",
    "INT32": "int32",
    "UINT32": "uint32",
    "INT64": "int64",
    "UINT64": "uint64",
    "FLOAT8_E4M3FN": "float8_e4m3fn",
    "FLOAT8_E5M2": "float8_e5m2",
    "BFLOAT16": "bfloat16",
    "FLOAT16": "float16",
    "FLOAT32": "float32",
    "FLOAT64": "float64",
    "COMPLEX64": "complex64",
    "COMPLEX128": "complex128",
    "PSTRING": "pstring",
}
