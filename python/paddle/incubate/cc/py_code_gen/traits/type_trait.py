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

from paddle.incubate.cc.py_code_gen.ir import ir_type


class TypeTrait:

    def t_null(self):
        return ir_type.NullType()

    def t_vec(self, *args):
        return ir_type.VectorType(args)

    def t_dtensor(self, shape, dtype):
        return ir_type.DenseTensorType(shape, dtype)

    def t_selected_rows(self):
        return ir_type.SelectedRowsType()

    def t_dense_tensor_array(self):
        return ir_type.DenseTensorArrayType()

    def t_sparse_coo_tensor(self):
        return ir_type.SparseCooTensorType()

    def t_sparse_csr_tensor(self):
        return ir_type.SparseCsrTensorType()

    def t_bf16(self):
        return ir_type.BFloat16Type()

    def t_f16(self):
        return ir_type.Float16Type()

    def t_f32(self):
        return ir_type.Float32Type()

    def t_f64(self):
        return ir_type.Float64Type()

    def t_i8(self):
        return ir_type.Int8Type()

    def t_ui8(self):
        return ir_type.UInt8Type()

    def t_i16(self):
        return ir_type.Int16Type()

    def t_i32(self):
        return ir_type.Int32Type()

    def t_i64(self):
        return ir_type.Int64Type()

    def t_index(self):
        return ir_type.IndexType()

    def t_bool(self):
        return ir_type.BoolType()

    def t_c64(self):
        return ir_type.Complex64Type()

    def t_c128(self):
        return ir_type.Complex128Type()

    def UnclassifiedType(self, *args, **kwargs):
        return ir_type.UnclassifiedType(args, kwargs)
