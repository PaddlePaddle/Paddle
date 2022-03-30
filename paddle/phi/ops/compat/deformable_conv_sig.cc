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

KernelSignature DeformableConvOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("deformable_conv",
                         {"Input", "Offset", "Filter", "Mask"},
                         {"strides",
                          "paddings",
                          "dilations",
                          "deformable_groups",
                          "groups",
                          "im2col_step"},
                         {"Output"});
}

KernelSignature DeformableConvGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "deformable_conv_grad",
      {"Input", "Offset", "Filter", "Mask", GradVarName("Output")},
      {"strides",
       "paddings",
       "dilations",
       "deformable_groups",
       "groups",
       "im2col_step"},
      {GradVarName("Input"),
       GradVarName("Offset"),
       GradVarName("Filter"),
       GradVarName("Mask")});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(deformable_conv_v1, deformable_conv);
PD_REGISTER_BASE_KERNEL_NAME(deformable_conv_v1_grad, deformable_conv_grad);

PD_REGISTER_ARG_MAPPING_FN(deformable_conv,
                           phi::DeformableConvOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(deformable_conv_grad,
                           phi::DeformableConvGradOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(deformable_conv_v1,
                           phi::DeformableConvOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(deformable_conv_v1_grad,
                           phi::DeformableConvGradOpArgumentMapping);
