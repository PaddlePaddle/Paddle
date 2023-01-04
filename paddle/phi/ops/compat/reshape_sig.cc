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

KernelSignature ReshapeOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.HasOutput("XShape")) {
    if (ctx.InputSize("ShapeTensor") > 0) {
      return KernelSignature(
          "reshape_with_xshape", {"X"}, {"ShapeTensor"}, {"Out", "XShape"});
    } else if (ctx.HasInput("Shape")) {
      return KernelSignature(
          "reshape_with_xshape", {"X"}, {"Shape"}, {"Out", "XShape"});
    } else {
      return KernelSignature(
          "reshape_with_xshape", {"X"}, {"shape"}, {"Out", "XShape"});
    }
  } else {
    if (ctx.InputSize("ShapeTensor") > 0) {
      return KernelSignature("reshape", {"X"}, {"ShapeTensor"}, {"Out"});
    } else if (ctx.HasInput("Shape")) {
      return KernelSignature("reshape", {"X"}, {"Shape"}, {"Out"});
    } else {
      return KernelSignature("reshape", {"X"}, {"shape"}, {"Out"});
    }
  }
}

KernelSignature ReshapeGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("reshape_grad", {"Out@GRAD"}, {}, {"X@GRAD"});
}

KernelSignature ReshapeDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("reshape_double_grad", {"DOut", "DDX"}, {}, {"DDOut"});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(reshape2, reshape);
PD_REGISTER_BASE_KERNEL_NAME(reshape2_grad, reshape_grad);
PD_REGISTER_BASE_KERNEL_NAME(reshape2_grad_grad, reshape_double_grad);

PD_REGISTER_ARG_MAPPING_FN(reshape2, phi::ReshapeOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(reshape2_grad, phi::ReshapeGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(reshape2_grad_grad,
                           phi::ReshapeDoubleGradOpArgumentMapping);
