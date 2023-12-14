/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

KernelSignature FusedBatchNormAddActOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("fused_bn_add_activation",
                         {"X", "Z", "Scale", "Bias", "Mean", "Variance"},
                         {"momentum", "epsilon", "act_type"},
                         {"Y",
                          "MeanOut",
                          "VarianceOut",
                          "SavedMean",
                          "SavedVariance",
                          "ReserveSpace"});
}

KernelSignature FusedBatchNormAddActGradOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("fused_bn_add_activation_grad",
                         {"X",
                          "Scale",
                          "Bias",
                          "Y",
                          "SavedMean",
                          "SavedVariance",
                          "ReserveSpace",
                          "Y@GRAD"},
                         {"momentum", "epsilon", "act_type"},
                         {"X@GRAD", "Z@GRAD", "Scale@GRAD", "Bias@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(fused_bn_add_activation,
                           phi::FusedBatchNormAddActOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(fused_bn_add_activation_grad,
                           phi::FusedBatchNormAddActGradOpArgumentMapping);
