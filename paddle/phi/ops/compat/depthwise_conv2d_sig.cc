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
                          "data_format"},
                         {"Output"});
}

KernelSignature DepthwiseConv2dGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("depthwise_conv2d_grad",
                         {"Input", "Filter", "Output@GRAD"},
                         {"strides",
                          "paddings",
                          "padding_algorithm",
                          "groups",
                          "dilations",
                          "data_format"},
                         {"Input@GRAD", "Filter@GRAD"});
}

KernelSignature DepthwiseConv2dDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
<<<<<<< HEAD
  return KernelSignature("depthwise_conv2d_grad_grad",
=======
  return KernelSignature("depthwise_conv2d_double_grad",
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
                         {"Input", "Filter", "DOutput", "DDInput", "DDFilter"},
                         {"strides",
                          "paddings",
                          "padding_algorithm",
                          "groups",
                          "dilations",
<<<<<<< HEAD
                          "data_format",
                          "use_addto",
                          "workspace_size_MB",
                          "exhaustive_search",
                          "fuse_relu_before_depthwise_conv"},
=======
                          "data_format"},
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
                         {"DInput", "DFilter", "DDOutput"});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(depthwise_conv2d_grad_grad,
                             depthwise_conv2d_double_grad);

PD_REGISTER_ARG_MAPPING_FN(depthwise_conv2d,
                           phi::DepthwiseConv2dOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(depthwise_conv2d_grad,
                           phi::DepthwiseConv2dGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(depthwise_conv2d_grad_grad,
                           phi::DepthwiseConv2dDoubleGradOpArgumentMapping);
