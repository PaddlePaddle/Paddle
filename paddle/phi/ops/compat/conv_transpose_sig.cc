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

KernelSignature Conv2dTransposeOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("conv2d_transpose",
                         {"Input", "Filter"},
                         {"strides",
                          "paddings",
                          "output_padding",
                          "output_size",
                          "padding_algorithm",
                          "groups",
                          "dilations",
                          "data_format"},
                         {"Output"});
}

KernelSignature Conv2dTransposeGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("conv2d_transpose_grad",
                         {"Input", "Filter", GradVarName("Output")},
                         {"strides",
                          "paddings",
                          "output_padding",
                          "output_size",
                          "padding_algorithm",
                          "groups",
                          "dilations",
                          "data_format"},
                         {GradVarName("Input"), GradVarName("Filter")});
}

KernelSignature Conv2dTransposeDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("conv2d_transpose_grad_grad",
                         {"Input", "Filter", "DOutput", "DDInput", "DDFilter"},
                         {"strides",
                          "paddings",
                          "output_padding",
                          "output_size",
                          "padding_algorithm",
                          "groups",
                          "dilations",
                          "data_format"},
                         {"DInput", "DFilter", "DDOutput"});
}

KernelSignature Conv3dTransposeOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("conv3d_transpose",
                         {"Input", "Filter"},
                         {"strides",
                          "paddings",
                          "output_padding",
                          "output_size",
                          "padding_algorithm",
                          "groups",
                          "dilations",
                          "data_format"},
                         {"Output"});
}

KernelSignature Conv3dTransposeGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("conv3d_transpose_grad",
                         {"Input", "Filter", GradVarName("Output")},
                         {"strides",
                          "paddings",
                          "output_padding",
                          "output_size",
                          "padding_algorithm",
                          "groups",
                          "dilations",
                          "data_format"},
                         {GradVarName("Input"), GradVarName("Filter")});
}

KernelSignature DepthwiseConv2dTransposeOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("depthwise_conv2d_transpose",
                         {"Input", "Filter"},
                         {"strides",
                          "paddings",
                          "output_padding",
                          "output_size",
                          "padding_algorithm",
                          "groups",
                          "dilations",
                          "data_format"},
                         {"Output"});
}

KernelSignature DepthwiseConv2dTransposeGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("depthwise_conv2d_transpose_grad",
                         {"Input", "Filter", GradVarName("Output")},
                         {"strides",
                          "paddings",
                          "output_padding",
                          "output_size",
                          "padding_algorithm",
                          "groups",
                          "dilations",
                          "data_format"},
                         {GradVarName("Input"), GradVarName("Filter")});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(conv2d_transpose,
                           phi::Conv2dTransposeOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(conv2d_transpose_grad,
                           phi::Conv2dTransposeGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(conv2d_transpose_grad_grad,
                           phi::Conv2dTransposeDoubleGradOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(conv3d_transpose,
                           phi::Conv3dTransposeOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(conv3d_transpose_grad,
                           phi::Conv3dTransposeGradOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(depthwise_conv2d_transpose,
                           phi::DepthwiseConv2dTransposeOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(depthwise_conv2d_transpose_grad,
                           phi::DepthwiseConv2dTransposeGradOpArgumentMapping);
