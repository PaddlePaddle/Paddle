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

KernelSignature MeanOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("mean_all", {"X"}, {}, {"Out"});
}

KernelSignature MeanGradOpGradArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "mean_all_grad", {"X", GradVarName("Out")}, {}, {GradVarName("X")});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(mean, mean_all);
PD_REGISTER_BASE_KERNEL_NAME(mean_grad, mean_all_grad);

PD_REGISTER_ARG_MAPPING_FN(mean, phi::MeanOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(mean_grad, phi::MeanGradOpGradArgumentMapping);
