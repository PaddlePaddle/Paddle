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

KernelSignature Conv2dOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (!ctx.HasAttr("use_addto") || !ctx.HasAttr("workspace_size_MB") ||
      !ctx.HasAttr("exhaustive_search")) {
    return KernelSignature("conv2d_infer",
                           {"Input", "Filter"},
                           {"strides",
                            "paddings",
                            "padding_algorithm",
                            "groups",
                            "dilations",
                            "data_format"},
                           {"Output"});
  } else {
    return KernelSignature("conv2d",
                           {"Input", "Filter"},
                           {"strides",
                            "paddings",
                            "padding_algorithm",
                            "groups",
                            "dilations",
                            "data_format",
                            "use_addto",
                            "workspace_size_MB",
                            "exhaustive_search"},
                           {"Output"});
  }
}

KernelSignature Conv2dGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("conv2d_grad",
                         {"Input", "Filter", "Output@GRAD"},
                         {"strides",
                          "paddings",
                          "padding_algorithm",
                          "groups",
                          "dilations",
                          "data_format",
                          "use_addto",
                          "workspace_size_MB",
                          "exhaustive_search"},
                         {"Input@GRAD", "Filter@GRAD"});
}

KernelSignature Conv2dDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("conv2d_grad_grad",
                         {"Input", "Filter", "DOutput", "DDInput", "DDFilter"},
                         {"strides",
                          "paddings",
                          "padding_algorithm",
                          "groups",
                          "dilations",
                          "data_format",
                          "use_addto",
                          "workspace_size_MB",
                          "exhaustive_search"},
                         {"DInput", "DFilter", "DDOutput"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(conv2d, phi::Conv2dOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(conv2d_grad, phi::Conv2dGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(conv2d_grad_grad,
                           phi::Conv2dDoubleGradOpArgumentMapping);
