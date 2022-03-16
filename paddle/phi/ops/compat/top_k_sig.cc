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

KernelSignature TopkOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("K")) {
    return KernelSignature(
        "top_k", {"X"}, {"K", "axis", "largest", "sorted"}, {"Out", "Indices"});

  } else {
    return KernelSignature(
        "top_k", {"X"}, {"k", "axis", "largest", "sorted"}, {"Out", "Indices"});
  }
}

KernelSignature TopkGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("top_k_grad",
                         {GradVarName("Out"), "X", "Indices"},
                         {"k", "axis", "largest", "sorted"},
                         {GradVarName("X")});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(top_k_v2, top_k);
PD_REGISTER_BASE_KERNEL_NAME(top_k_v2_grad, top_k_grad);
PD_REGISTER_ARG_MAPPING_FN(top_k_v2, phi::TopkOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(top_k_v2_grad, phi::TopkGradOpArgumentMapping);
