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

KernelSignature DepthwiseConv2dOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("depthwise_conv2d",
                         {"Input", "Filter"},
                         {"strides",
                          "paddings",
                          "padding_algorithm",
                          "groups",
                          "dilations",
                          "data_format",
                          "use_addto",
                          "workspace_size_MB",
                          "exhaustive_search",
                          "fuse_relu_before_depthwise_conv"},
                         {"Output"});
}

KernelSignature DepthwiseConv2dGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("depthwise_conv2d_grad",
                         {GradVarName("Output"), "Input", "Filter"},
                         {"strides",
                          "paddings",
                          "padding_algorithm",
                          "groups",
                          "dilations",
                          "data_format",
                          "use_addto",
                          "workspace_size_MB",
                          "exhaustive_search",
                          "fuse_relu_before_depthwise_conv"},
                         {GradVarName("Input"), GradVarName("Filter")});
}

KernelSignature DepthwiseConv2dDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("depthwise_conv2d_grad_grad",
                         {"DDInput", "DDFilter", "DOutput", "Input", "Filter"},
                         {"strides",
                          "paddings",
                          "padding_algorithm",
                          "groups",
                          "dilations",
                          "data_format",
                          "use_addto",
                          "workspace_size_MB",
                          "exhaustive_search",
                          "fuse_relu_before_depthwise_conv"},
                         {"DDOutput", "DInput", "DFilter"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(depthwise_conv2d,
                           phi::DepthwiseConv2dOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(depthwise_conv2d_grad,
                           phi::DepthwiseConv2dGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(depthwise_conv2d_grad_grad,
                           phi::DepthwiseConv2dDoubleGradOpArgumentMapping);
