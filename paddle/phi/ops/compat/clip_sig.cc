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
#include "paddle/utils/small_vector.h"

namespace phi {

KernelSignature ClipOpArgumentMapping(const ArgumentMappingContext& ctx) {
  paddle::SmallVector<std::string> attr_names;
  attr_names.emplace_back(ctx.HasInput("Min") ? "Min" : "min");
  attr_names.emplace_back(ctx.HasInput("Max") ? "Max" : "max");
  if (ctx.IsDenseTensorInput("X")) {
    return KernelSignature("clip", {"X"}, std::move(attr_names), {"Out"});
  } else if (ctx.IsSelectedRowsInput("X")) {
    return KernelSignature(
        "clip_dense_param_sparse_grad", {"X"}, std::move(attr_names), {"Out"});
  }

  return KernelSignature("unregistered", {}, {}, {});
}

KernelSignature ClipGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  paddle::SmallVector<std::string> attr_names;
  attr_names.emplace_back(ctx.HasInput("Min") ? "Min" : "min");
  attr_names.emplace_back(ctx.HasInput("Max") ? "Max" : "max");
  return KernelSignature("clip_grad",
                         {"X", GradVarName("Out")},
                         std::move(attr_names),
                         {GradVarName("X")});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(clip, phi::ClipOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(clip_grad, phi::ClipGradOpArgumentMapping);
