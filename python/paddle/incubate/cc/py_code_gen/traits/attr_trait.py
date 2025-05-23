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

from paddle.incubate.cc.py_code_gen.ir import ir_attr


class AttrTrait:

    def a_bool(self, value):
        return ir_attr.BoolAttribute(value)

    def a_c64(self, real, imag):
        return ir_attr.Complex64Attribute(real, imag)

    def a_c128(self, real, imag):
        return ir_attr.Complex128Attribute(real, imag)

    def a_f32(self, value):
        return ir_attr.Float32Attribute(value)

    def a_f64(self, value):
        return ir_attr.Float64Attribute(value)

    def a_i32(self, value):
        return ir_attr.Int32Attribute(value)

    def a_index(self, value):
        return ir_attr.IndexAttribute(value)

    def a_i64(self, value):
        return ir_attr.Int64Attribute(value)

    def a_pointer(self, value):
        return ir_attr.PointerAttribute(value)

    def a_type(self, value):
        return ir_attr.TypeAttribute(value)

    def a_str(self, value):
        return ir_attr.StrAttribute(value)

    def a_array(self, *value):
        return ir_attr.ArrayAttribute(value)

    def a_tensorname(self, value):
        return ir_attr.TensorNameAttribute(value)

    def a_intarray(self, *value):
        return ir_attr.IntArrayAttribute(value)

    def a_scalar(self, value):
        return ir_attr.ScalarAttribute(value)

    def a_dtype(self, dtype_name):
        if dtype_name == "Undefined":
            dtype_name = None
        return ir_attr.DataTypeAttribute(dtype_name)

    def a_place(self, type, device=None):
        return ir_attr.PlaceAttribute(type, device)

    def a_layout(self, name):
        return ir_attr.DataLayoutAttribute(name)

    def a_kernel(self, value=None):
        return ir_attr.KernelAttribute(value)

    def a_group_info(self, value=None):
        return ir_attr.GroupInfoAttribute(value)

    def a_cinn_kernel_info(self, value=None):
        return ir_attr.CINNKernelInfoAttribute(value)

    def a_symbol(self, value=None):
        return ir_attr.SymbolAttribute(value)

    def UnclassifiedAttribute(self, value=None):
        return ir_attr.UnclassifiedAttribute(value)
