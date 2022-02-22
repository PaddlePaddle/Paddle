/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature MatmulGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.HasAttr("use_addto")) {
    return KernelSignature("addto_matmul_grad",
                           {"X", "Y", GradVarName("Out")},
                           {"trans_x", "trans_y", "use_addto"},
                           {GradVarName("X"), GradVarName("Y")});
  } else {
    return KernelSignature("matmul_grad",
                           {"X", "Y", GradVarName("Out")},
                           {"trans_x", "trans_y"},
                           {GradVarName("X"), GradVarName("Y")});
  }
}

KernelSignature MatmulDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("matmul_double_grad",
                         {"X", "Y", "DOut", "DDX", "DDY"},
                         {"trans_x", "trans_y"},
                         {"DX", "DY", "DDOut"});
}

KernelSignature MatmulTripleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "matmul_triple_grad",
      {"X", "Y", "DOut", "DDX", "DDY", "D_DX", "D_DY", "D_DDOut"},
      {"trans_x", "trans_y"},
      {"D_X_out", "D_Y_out", "D_DOut_out", "D_DDX_out", "D_DDY_out"});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(matmul_v2, matmul);
PD_REGISTER_BASE_KERNEL_NAME(matmul_v2_grad, matmul_grad);
PD_REGISTER_BASE_KERNEL_NAME(matmul_v2_grad_grad, matmul_double_grad);
PD_REGISTER_BASE_KERNEL_NAME(matmul_v2_triple_grad, matmul_triple_grad);

PD_REGISTER_ARG_MAPPING_FN(matmul_v2_grad, phi::MatmulGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(matmul_v2_grad_grad,
                           phi::MatmulDoubleGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(matmul_v2_triple_grad,
                           phi::MatmulTripleGradOpArgumentMapping);
