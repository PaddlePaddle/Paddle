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


class KernelArgIdNameRegistry:

    def __init__(self, code_gen_ctx, tensor_match_ctx, name_prefix):
        self.code_gen_ctx = code_gen_ctx
        self.tensor_match_ctx = tensor_match_ctx
        self.name_prefix = name_prefix
        self.generated_kernel_arg_id2unique_name = ap.MutableOrderedDict()
        self.all_kernel_arg_id2unique_name = ap.MutableOrderedDict()
        self.in_tensor_data_ptr_seq_no = 0
        self.out_tensor_data_ptr_seq_no = 0
        self.dim_expr_seq_no = 0

    def get_or_create_kernel_arg_id_manul_var_name(
        self, kernel_arg_id, cpp_var_name
    ):
        create = lambda: cpp_var_name
        return self.all_kernel_arg_id2unique_name.get_or_create(
            kernel_arg_id, create
        )

    def get_in_tensor_data_ptr_var_name(self, in_ir_value_name):
        ir_value = getattr(self.tensor_match_ctx, in_ir_value_name)
        kernel_arg_id = self.code_gen_ctx.in_tensor_data_ptr_kernel_arg_id(
            ir_value
        )
        create = self._get_creator(
            kernel_arg_id, self._create_in_tensor_data_ptr_var_name
        )
        return self.generated_kernel_arg_id2unique_name.get_or_create(
            kernel_arg_id, create
        )

    def _get_creator(self, kernel_arg_id, backend_creator):
        return lambda: self.all_kernel_arg_id2unique_name.get_or_create(
            kernel_arg_id, backend_creator
        )

    def _create_in_tensor_data_ptr_var_name(self):
        name = f"{self.name_prefix}in_ptr_{self.in_tensor_data_ptr_seq_no}"
        self.in_tensor_data_ptr_seq_no = self.in_tensor_data_ptr_seq_no + 1
        return name

    def get_out_tensor_data_ptr_var_name(self, out_ir_value_name):
        ir_value = getattr(self.tensor_match_ctx, out_ir_value_name)
        kernel_arg_id = self.code_gen_ctx.out_tensor_data_ptr_kernel_arg_id(
            ir_value
        )
        create = self._get_creator(
            kernel_arg_id, self._create_out_tensor_data_ptr_var_name
        )
        return self.generated_kernel_arg_id2unique_name.get_or_create(
            kernel_arg_id, create
        )

    def _create_out_tensor_data_ptr_var_name(self):
        name = f"{self.name_prefix}out_ptr_{self.out_tensor_data_ptr_seq_no}"
        self.out_tensor_data_ptr_seq_no = self.out_tensor_data_ptr_seq_no + 1
        return name

    def get_dim_expr_var_name(self, dim_expr):
        kernel_arg_id = self.code_gen_ctx.dim_expr_kernel_arg_id(dim_expr)
        create = self._get_creator(
            kernel_arg_id, self._create_dim_expr_var_name
        )
        return self.generated_kernel_arg_id2unique_name.get_or_create(
            kernel_arg_id, create
        )

    def _create_dim_expr_var_name(self):
        name = f"{self.name_prefix}dim_{self.dim_expr_seq_no}"
        self.dim_expr_seq_no = self.dim_expr_seq_no + 1
        return name
