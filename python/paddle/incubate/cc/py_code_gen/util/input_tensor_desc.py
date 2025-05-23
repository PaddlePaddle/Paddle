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

from collections import namedtuple

InputTensorDesc = namedtuple(
    "InputTensorDesc",
    [
        "shape",
        "dtype",
        "big_dtype",
        "data",
        "min",
        "max",
    ],
)


def MakeInputTensorDesc(shape, dtype, data):
    return InputTensorDesc(
        shape=shape,
        dtype=dtype,
        big_dtype=_GetBigType(dtype),
        data=data,
        min=getattr(InitMinGetter, dtype)(),
        max=getattr(InitMaxGetter, dtype)(),
    )


def _GetBigType(dtype):
    return type2bigger_type[dtype]


type2bigger_type = {
    "bool": "bool",
    "bfloat16": "float64",
    "float16": "float64",
    "float32": "float64",
    "float64": "float64",
    "int8": "int64",
    "int16": "int64",
    "int32": "int64",
    "int64": "int64",
    "uint8": "int64",
    "uint16": "int64",
    "uint32": "int64",
    "uint64": "int64",
}


class InitMinGetter:
    def bool():
        return "False"

    def bfloat16():
        return "0"

    def float16():
        return "0"

    def float32():
        return "0"

    def float64():
        return "0"

    def int8():
        return "0"

    def int16():
        return "0"

    def int32():
        return "0"

    def int64():
        return "0"

    def uint8():
        return "0"

    def uint16():
        return "0"

    def uint32():
        return "0"

    def uint64():
        return "0"


class InitMaxGetter:
    def bool():
        return "True"

    def bfloat16():
        return "0.5"

    def float16():
        return "0.5"

    def float32():
        return "0.5"

    def float64():
        return "0.5"

    def int8():
        return "3"

    def int16():
        return "3"

    def int32():
        return "3"

    def int64():
        return "3"

    def uint8():
        return "3"

    def uint16():
        return "3"

    def uint32():
        return "3"

    def uint64():
        return "3"
