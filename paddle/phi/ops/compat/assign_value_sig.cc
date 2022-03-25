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

KernelSignature AssignValueOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  // if use `dtype`. here will depend the fluid proto
  if (ctx.HasAttr("bool_values") &&
      !paddle::any_cast<std::vector<int>>(ctx.Attr("bool_values")).empty()) {
    return KernelSignature(
        "assign_value", {}, {"shape", "dtype", "bool_values"}, {"Out"});
  } else if (ctx.HasAttr("fp32_values") &&
             !paddle::any_cast<std::vector<float>>(ctx.Attr("fp32_values"))
                  .empty()) {
    return KernelSignature(
        "assign_value", {}, {"shape", "dtype", "fp32_values"}, {"Out"});
  } else if (ctx.HasAttr("int32_values") &&
             !paddle::any_cast<std::vector<int>>(ctx.Attr("int32_values"))
                  .empty()) {
    return KernelSignature(
        "assign_value", {}, {"shape", "dtype", "int32_values"}, {"Out"});
  } else if (ctx.HasAttr("int64_values") &&
             !paddle::any_cast<std::vector<int64_t>>(ctx.Attr("int64_values"))
                  .empty()) {
    return KernelSignature(
        "assign_value", {}, {"shape", "dtype", "int64_values"}, {"Out"});
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(assign_value, phi::AssignValueOpArgumentMapping);
