/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/core/compat/op_utils.h"

namespace pten {

KernelSignature FlattenOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.HasOutput("XShape")) {
    return KernelSignature("flatten_with_xshape",
                           {"X"},
                           {"start_axis", "stop_axis"},
                           {"Out", "XShape"});
  } else {
    return KernelSignature(
        "flatten", {"X"}, {"start_axis", "stop_axis"}, {"Out"});
  }
}

KernelSignature FlattenGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "flatten_grad", {GradVarName("Out"), "XShape"}, {}, {GradVarName("X")});
}

}  // namespace pten

PT_REGISTER_BASE_KERNEL_NAME(flatten_contiguous_range, flatten);
PT_REGISTER_BASE_KERNEL_NAME(flatten_contiguous_range_grad, flatten_grad);

PT_REGISTER_ARG_MAPPING_FN(flatten_contiguous_range,
                           pten::FlattenOpArgumentMapping);
PT_REGISTER_ARG_MAPPING_FN(flatten_contiguous_range_grad,
                           pten::FlattenGradOpArgumentMapping);
