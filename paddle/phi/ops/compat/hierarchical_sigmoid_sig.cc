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

KernelSignature HierarchicalSigmoidOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("hsigmoid_loss",
                         {"X", "Label", "W", "Bias", "PathTable", "PathCode"},
                         {"num_classes", "is_sparse"},
                         {"Out", "PreOut", "W_Out"});
}

KernelSignature HierarchicalSigmoidGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsDenseTensorOutput("W@GRAD")) {
    return KernelSignature("hsigmoid_loss_grad",
                           {"X",
                            "W",
                            "Label",
                            "PathTable",
                            "PathCode",
                            "Bias",
                            "PreOut",
                            "Out@GRAD"},
                           {"num_classes", "is_sparse"},
                           {"X@GRAD", "W@GRAD", "Bias@GRAD"});
  } else if (ctx.IsSelectedRowsOutput("W@GRAD")) {
    return KernelSignature("hsigmoid_loss_grad_sr",
                           {"X",
                            "W",
                            "Label",
                            "PathTable",
                            "PathCode",
                            "Bias",
                            "PreOut",
                            "Out@GRAD"},
                           {"num_classes", "is_sparse"},
                           {"X@GRAD", "W@GRAD", "Bias@GRAD"});
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(hierarchical_sigmoid, hsigmoid_loss);
PD_REGISTER_BASE_KERNEL_NAME(hierarchical_sigmoid_grad, hsigmoid_loss_grad);

PD_REGISTER_ARG_MAPPING_FN(hierarchical_sigmoid,
                           phi::HierarchicalSigmoidOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(hierarchical_sigmoid_grad,
                           phi::HierarchicalSigmoidGradOpArgumentMapping);
