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
  int axis = paddle::any_cast<int>(ctx.Attr("axis"));
  if (axis == -1) {
    return KernelSignature("fused_add",
                           {"X", "Y"},
                           {"fuse_alpha",
                            "fuse_beta",
                            "fuse_activation",
                            "mkldnn_data_type",
                            "scale_x",
                            "scale_y",
                            "scale_out",
                            "fused_output_scale"},
                           {"Out"});
  }
  return KernelSignature("fused_add_raw",
                         {"X", "Y"},
                         {"axis",
                          "fuse_alpha",
                          "fuse_beta",
                          "fuse_activation",
                          "mkldnn_data_type",
                          "scale_x",
                          "scale_y",
                          "scale_out",
                          "fused_output_scale"},
                         {"Out"});
}

KernelSignature FusedElementwiseSubOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  int axis = paddle::any_cast<int>(ctx.Attr("axis"));
  if (axis == -1) {
    return KernelSignature("fused_subtract",
                           {"X", "Y"},
                           {"fuse_alpha",
                            "fuse_beta",
                            "fuse_activation",
                            "mkldnn_data_type",
                            "scale_x",
                            "scale_y",
                            "scale_out",
                            "fused_output_scale"},
                           {"Out"});
  }
  return KernelSignature("fused_subtract_raw",
                         {"X", "Y"},
                         {"axis",
                          "fuse_alpha",
                          "fuse_beta",
                          "fuse_activation",
                          "mkldnn_data_type",
                          "scale_x",
                          "scale_y",
                          "scale_out",
                          "fused_output_scale"},
                         {"Out"});
}

KernelSignature FusedElementwiseMulOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  int axis = paddle::any_cast<int>(ctx.Attr("axis"));
  if (axis == -1) {
    return KernelSignature("fused_multiply",
                           {"X", "Y"},
                           {"fuse_alpha",
                            "fuse_beta",
                            "fuse_activation",
                            "mkldnn_data_type",
                            "scale_x",
                            "scale_y",
                            "scale_out",
                            "fused_output_scale",
                            "fused_unsqueeze2_axes"},
                           {"Out"});
  }
  return KernelSignature("fused_multiply_raw",
                         {"X", "Y"},
                         {"axis",
                          "fuse_alpha",
                          "fuse_beta",
                          "fuse_activation",
                          "mkldnn_data_type",
                          "scale_x",
                          "scale_y",
                          "scale_out",
                          "fused_output_scale",
                          "fused_unsqueeze2_axes"},
                         {"Out"});
}

KernelSignature FusedElementwiseDivOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  int axis = paddle::any_cast<int>(ctx.Attr("axis"));
  if (axis == -1) {
    return KernelSignature("fused_divide",
                           {"X", "Y"},
                           {"fuse_alpha",
                            "fuse_beta",
                            "fuse_activation",
                            "mkldnn_data_type",
                            "scale_x",
                            "scale_y",
                            "scale_out",
                            "fused_output_scale"},
                           {"Out"});
  }
  return KernelSignature("fused_divide_raw",
                         {"X", "Y"},
                         {"axis",
                          "fuse_fuse_alpha",
                          "fuse_beta",
                          "fuse_activation",
                          "mkldnn_data_type",
                          "scale_x",
                          "scale_y",
                          "scale_out",
                          "fused_output_scale"},
                         {"Out"});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(fused_elementwise_add, fused_add);
PD_REGISTER_BASE_KERNEL_NAME(fused_elementwise_sub, fused_subtract);
PD_REGISTER_BASE_KERNEL_NAME(fused_elementwise_mul, fused_multiply);
PD_REGISTER_BASE_KERNEL_NAME(fused_elementwise_div, fused_divide);

PD_REGISTER_ARG_MAPPING_FN(fused_elementwise_add,
                           phi::FusedElementwiseAddOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(fused_elementwise_sub,
                           phi::FusedElementwiseSubOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(fused_elementwise_mul,
                           phi::FusedElementwiseMulOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(fused_elementwise_div,
                           phi::FusedElementwiseDivOpArgumentMapping);
