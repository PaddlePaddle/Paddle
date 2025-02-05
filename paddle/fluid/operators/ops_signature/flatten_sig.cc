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

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature FlattenOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.IsForInferShape()) {
    return KernelSignature(
        "flatten", {"X"}, {"start_axis", "stop_axis"}, {"Out", "XShape"});
  }
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
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature(
      "flatten_grad", {"XShape", "Out@GRAD"}, {}, {"X@GRAD"});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(flatten_contiguous_range, flatten);
PD_REGISTER_BASE_KERNEL_NAME(flatten_contiguous_range, flatten_with_xshape);

PD_REGISTER_ARG_MAPPING_FN(flatten_contiguous_range,
                           phi::FlattenOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(flatten_contiguous_range_grad,
                           phi::FlattenGradOpArgumentMapping);
