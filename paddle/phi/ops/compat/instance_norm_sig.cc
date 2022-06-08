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

KernelSignature InstanceNormOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("instance_norm",
                         {"X", "Scale", "Bias"},
                         {"epsilon"},
                         {"Y", "SavedMean", "SavedVariance"});
}

KernelSignature InstanceNormGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("instance_norm_grad",
                         {"X", "Scale", "SavedMean", "SavedVariance", "Y@GRAD"},
                         {"epsilon"},
                         {"X@GRAD", "Scale@GRAD", "Bias@GRAD"});
}
KernelSignature InstanceNormDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("instance_norm_double_grad",
                         {"X",
                          "Scale",
                          "SavedMean",
                          "SavedVariance",
                          "DY",
                          "DDX",
                          "DDScale",
                          "DDBias"},
                         {"epsilon"},
                         {"DX", "DScale", "DDY"});
}
}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(instance_norm_grad_grad,
                             instance_norm_double_grad);
PD_REGISTER_ARG_MAPPING_FN(instance_norm, phi::InstanceNormOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(instance_norm_grad,
                           phi::InstanceNormGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(instance_norm_grad_grad,
                           phi::InstanceNormDoubleGradOpArgumentMapping);
