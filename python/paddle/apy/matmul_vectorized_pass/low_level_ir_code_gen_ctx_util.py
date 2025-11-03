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

import ap


class CudaLikeIrCodeGenCtx:
    def __init__(self, compute_dtype):
        self.stmts = ap.MutableList()
        self.dtype2type_name = ap.OrderedDict(
            [
                [ap.DataType.float, "float"],
                [ap.DataType.float16, "half"],
                [ap.DataType.bfloat16, "nv_bfloat16"],
                [ap.DataType.int32, "int"],
                [ap.DataType.int64, "int64_t"],
            ]
        )
        self.compute_dtype = compute_dtype
        self.compute_dtype_name = self.dtype2type_name[self.compute_dtype]
        self.type_cast_str_list = [
            "",
            f"static_cast<{self.compute_dtype_name}>",
        ]

    def assign(self, dst, src):
        self.stmts.append(f"{dst.var_name} = {src.var_name};")

    def let(self, var, val_name):
        var_dtype_name = self.dtype2type_name[var.get_dtype()]
        is_same = self.compute_dtype == var.get_dtype()
        type_name = (
            f"{var_dtype_name}" if is_same else f"{self.compute_dtype_name}"
        )
        type_cast_str = (
            "" if is_same else f"static_cast<{self.compute_dtype_name}>"
        )
        self.stmts.append(
            f"{type_name} {var.var_name} = {type_cast_str}({val_name});"
        )

    def store(self, dtype, dst, offset_var_name, src):
        is_same = dtype == self.dtype2type_name[self.compute_dtype]
        type_cast_str = "" if is_same else f"static_cast<{dtype}>"
        self.stmts.append(f"{dst}[{offset_var_name}] = {type_cast_str}({src});")

    def get_stmts_joined_str(self, indent):
        return f"\n{indent}".join([*self.stmts])


class CudaLikeVectorizedIrCodeGenCtx:
    def __init__(self, compute_dtype):
        self.load_vector_stmts = ap.MutableList()
        self.unroll_stmts = ap.MutableList()
        self.dtype2type_name = ap.OrderedDict(
            [
                [ap.DataType.float, "float"],
                [ap.DataType.float16, "half"],
                [ap.DataType.bfloat16, "nv_bfloat16"],
                [ap.DataType.int32, "int"],
                [ap.DataType.int64, "int64_t"],
            ]
        )
        self.compute_dtype = compute_dtype
        self.compute_dtype_name = self.dtype2type_name[self.compute_dtype]

    def assign(self, dst, src):
        self.unroll_stmts.append(f"{dst.var_name} = {src.var_name};")
    
    def let(self, var, val_name):
        var_dtype_name = self.dtype2type_name[var.get_dtype()]
        is_same = self.compute_dtype == var.get_dtype()
        type_name = (
            f"{var_dtype_name}" if is_same else f"{self.compute_dtype_name}"
        )
        type_cast_str = (
            "" if is_same else f"static_cast<{self.compute_dtype_name}>"
        )
        self.unroll_stmts.append(
            f"{type_name} {var.var_name} = {type_cast_str}({val_name});"
        )

    def load_vector(self, scalar_var, ptr_var_name, offset_var_name, size_var_name):
        scalar_type_name = self.dtype2type_name[scalar_var.get_dtype()]
        vector_var_name = f"{scalar_var.var_name}_vec"
        self.load_vector_stmts.append(f"InVectorType<{scalar_type_name}> {vector_var_name} = load_vector<{scalar_type_name}, VecSize>({ptr_var_name}, {offset_var_name}, coord.is_valid, {size_var_name});")
        
        self.let(scalar_var, f"extract_scalar({vector_var_name}, i)")

    def get_load_vector_stmts_joined_str(self, indent):
        return f"\n{indent}".join([*self.load_vector_stmts])

    def get_unroll_stmts_joined_str(self, indent):
        return f"\n{indent}".join([*self.unroll_stmts])


    # def assign_vec_element(self, dst_vec, src):
    #     self.assign(f"{dst_vec}(i)", src)