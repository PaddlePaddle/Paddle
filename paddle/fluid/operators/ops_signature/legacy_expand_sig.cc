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

KernelSignature LegacyExpandOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  const auto& shape =
      paddle::any_cast<std::vector<int>>(ctx.Attr("expand_times"));
  // Infer output shape by Attr("shape") in CompileTime if it is specified.
  if (!ctx.IsRuntime() && !shape.empty()) {
    return KernelSignature("legacy_expand", {"X"}, {"expand_times"}, {"Out"});
  }
  if (ctx.HasInput("ExpandTimes")) {
    return KernelSignature("legacy_expand", {"X"}, {"ExpandTimes"}, {"Out"});
  } else if (ctx.InputSize("expand_times_tensor") > 0) {
    return KernelSignature(
        "legacy_expand", {"X"}, {"expand_times_tensor"}, {"Out"});
  } else {
    return KernelSignature("legacy_expand", {"X"}, {"expand_times"}, {"Out"});
  }
}

KernelSignature LegacyExpandGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  const auto& shape =
      paddle::any_cast<std::vector<int>>(ctx.Attr("expand_times"));
  // Infer output shape by Attr("shape") in CompileTime if it is specified.
  if (!ctx.IsRuntime() && !shape.empty()) {
    return KernelSignature(
        "legacy_expand_grad", {"X", "Out@GRAD"}, {"expand_times"}, {"X@GRAD"});
  }
  if (ctx.HasInput("ExpandTimes")) {
    return KernelSignature(
        "legacy_expand_grad", {"X", "Out@GRAD"}, {"ExpandTimes"}, {"X@GRAD"});
  } else if (ctx.InputSize("expand_times_tensor") > 0) {
    return KernelSignature("legacy_expand_grad",
                           {"X", "Out@GRAD"},
                           {"expand_times_tensor"},
                           {"X@GRAD"});
  } else {
    return KernelSignature(
        "legacy_expand_grad", {"X", "Out@GRAD"}, {"expand_times"}, {"X@GRAD"});
  }
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(expand, legacy_expand);
PD_REGISTER_BASE_KERNEL_NAME(expand_grad, legacy_expand_grad);

PD_REGISTER_ARG_MAPPING_FN(expand, phi::LegacyExpandOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(expand_grad, phi::LegacyExpandGradOpArgumentMapping);
