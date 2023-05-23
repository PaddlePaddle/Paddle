// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

KernelSignature FusedElementwiseAddOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("fused_elementwise_add",
                         {"X", "Y"},
                         {"axis",
                          "fuse_activation",
                          "fuse_alpha",
                          "fuse_beta",
                          "fused_output_scale",
                          "fused_unsqueeze2_axes",
                          "scale_x",
                          "scale_y",
                          "scale_out"},
                         {"Out"});
}

KernelSignature FusedElementwiseSubOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("fused_elementwise_sub",
                         {"X", "Y"},
                         {"axis",
                          "fuse_activation",
                          "fuse_alpha",
                          "fuse_beta",
                          "fused_output_scale",
                          "fused_unsqueeze2_axes",
                          "scale_x",
                          "scale_y",
                          "scale_out"},
                         {"Out"});
}

KernelSignature FusedElementwiseMulOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("fused_elementwise_mul",
                         {"X", "Y"},
                         {"axis",
                          "fuse_activation",
                          "fuse_alpha",
                          "fuse_beta",
                          "fused_output_scale",
                          "fused_unsqueeze2_axes",
                          "scale_x",
                          "scale_y",
                          "scale_out"},
                         {"Out"});
}

KernelSignature FusedElementwiseDivOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("fused_elementwise_div",
                         {"X", "Y"},
                         {"axis",
                          "fuse_activation",
                          "fuse_alpha",
                          "fuse_beta",
                          "fused_output_scale",
                          "fused_unsqueeze2_axes",
                          "scale_x",
                          "scale_y",
                          "scale_out"},
                         {"Out"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(fused_elementwise_add,
                           phi::FusedElementwiseAddOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(fused_elementwise_sub,
                           phi::FusedElementwiseSubOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(fused_elementwise_mul,
                           phi::FusedElementwiseMulOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(fused_elementwise_div,
                           phi::FusedElementwiseDivOpArgumentMapping);
