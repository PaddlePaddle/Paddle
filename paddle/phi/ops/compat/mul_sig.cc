// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature MulGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("matmul_with_flatten_grad",
                         {"X", "Y", GradVarName("Out")},
                         {"x_num_col_dims", "y_num_col_dims"},
                         {GradVarName("X"), GradVarName("Y")});
}

KernelSignature MulDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("matmul_with_flatten_double_grad",
                         {"X", "Y", "DOut", "DDX", "DDY"},
                         {"x_num_col_dims", "y_num_col_dims"},
                         {"DX", "DY", "DDOut"});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(mul, matmul_with_flatten);
PD_REGISTER_BASE_KERNEL_NAME(mul_grad, matmul_with_flatten_grad);
PD_REGISTER_BASE_KERNEL_NAME(mul_grad_grad, matmul_with_flatten_double_grad);

PD_REGISTER_ARG_MAPPING_FN(mul_grad, phi::MulGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(mul_grad_grad, phi::MulDoubleGradOpArgumentMapping);
