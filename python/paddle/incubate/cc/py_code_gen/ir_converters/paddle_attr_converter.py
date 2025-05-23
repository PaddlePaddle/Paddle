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


def ConvertAttributeToString(attr):
    return getattr(AttributeToStringConverter, type(attr).__name__)(attr)


class AttributeToStringConverter:
    def BoolAttribute(attr):
        return str(attr.value)

    def Complex64Attribute(attr):
        return f"complex(float('{attr.real}'), float('{attr.imag}'))"

    def Complex128Attribute(attr):
        return f"complex(float('{attr.real}'), float('{attr.imag}'))"

    def Float32Attribute(attr):
        return f"float('{attr.value}')"

    def Float64Attribute(attr):
        return f"float('{attr.value}')"

    def Int32Attribute(attr):
        return str(attr.value)

    def IndexAttribute(attr):
        return str(attr.value)

    def Int64Attribute(attr):
        return str(attr.value)

    def PointerAttribute(attr):
        return f"'{attr.value}'"

    def TypeAttribute(attr):
        return f"'{attr.value}'"

    def StrAttribute(attr):
        return repr(attr.value)

    def ArrayAttribute(attr):
        return "[" + ", ".join(map(ConvertAttributeToString, attr.value)) + "]"

    def TensorNameAttribute(attr):
        return f"'{attr.value}'"

    def IntArrayAttribute(attr):
        return "[" + ", ".join(map(str, attr.value)) + "]"

    def ScalarAttribute(attr):
        raise NotImplementedError('ScalarAttribute Converter not implemented.')

    def DataTypeAttribute(attr):
        if attr.value is None:
            return "None"
        return f"paddle.{attr.value}"

    def PlaceAttribute(attr):
        if attr.type == "cpu":
            return "paddle.core.CPUPlace()"
        if attr.type == "undefined":
            return "paddle.framework._current_expected_place()"
        return f'"{attr.type}:{attr.device}"'

    def DataLayoutAttribute(attr):
        return f"'{attr.value}'"

    def KernelAttribute(attr):
        return "None"

    def GroupInfoAttribute(attr):
        return "None"

    def CINNKernelInfoAttribute(attr):
        return "None"

    def SymbolAttribute(attr):
        return "None"

    def UnclassifiedAttribute(attr):
        return "None"
