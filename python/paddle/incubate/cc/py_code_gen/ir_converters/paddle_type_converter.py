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


def ConvertTypeToString(t):
    return getattr(TypeToStringConverter, type(t).__name__)(t)


class TypeToStringConverter:
    def VectorType(t):
        return "[" + ", ".join(map(ConvertTypeToString, t.value)) + "]"

    def DenseTensorType(t):
        return (
            f"({ConvertTypeToString(t.shape)}, {ConvertTypeToString(t.dtype)})"
        )

    def BFloat16Type(t):
        return "bfloat16"

    def Float16Type(t):
        return "float16"

    def Float32Type(t):
        return "float32"

    def Float64Type(t):
        return "float64"

    def Int8Type(t):
        return "int8"

    def UInt8Type(t):
        return "uint8"

    def Int16Type(t):
        return "int16"

    def Int32Type(t):
        return "int32"

    def Int64Type(t):
        return "int64"

    def IndexType(t):
        return "index"

    def BoolType(t):
        return "bool"

    def Complex64Type(t):
        return "complex64"

    def Complex128Type(t):
        return "complex128"
