
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

KernelSignature UnsqueezeOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.InputSize("AxesTensorList") > 0) {
    VLOG(2) << "unsqueeze2 in AxesTensorList";
    return KernelSignature(
        "unsqueeze", {"X"}, {"AxesTensorList"}, {"XShape", "Out"});
  } else if (ctx.InputSize("AxesTensor") > 0) {
    VLOG(2) << "unsqueeze2 in AxesTensor";
    return KernelSignature(
        "unsqueeze", {"X"}, {"AxesTensor"}, {"XShape", "Out"});
  } else {
    VLOG(2) << "unsqueeze2 in axes";
    return KernelSignature("unsqueeze", {"X"}, {"axes"}, {"XShape", "Out"});
  }
}

KernelSignature UnsqueezeGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "unsqueeze_grad", {"XShape", "Out@GRAD"}, {}, {"X@GRAD"});
}
}  // namespace phi
PD_REGISTER_BASE_KERNEL_NAME(unsqueeze2, unsqueeze);
PD_REGISTER_BASE_KERNEL_NAME(unsqueeze2_grad, unsqueeze_grad);

PD_REGISTER_ARG_MAPPING_FN(unsqueeze2, phi::UnsqueezeOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(unsqueeze2_grad,
                           phi::UnsqueezeGradOpArgumentMapping);
