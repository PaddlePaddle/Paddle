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

KernelSignature SplitOpArgumentMapping(const ArgumentMappingContext& ctx) {
  // priority:  num > SectionsTensorList > sections
  // priority: AxisTensor > axis
  if (paddle::any_cast<int>(ctx.Attr("num")) > 0) {
    if (ctx.HasInput("AxisTensor")) {
      return KernelSignature(
          "split_with_num", {"X"}, {"num", "AxisTensor"}, {"Out"});
    } else {
      return KernelSignature("split_with_num", {"X"}, {"num", "axis"}, {"Out"});
    }
  }

  if (ctx.InputSize("SectionsTensorList") > 0) {
    if (ctx.HasInput("AxisTensor")) {
      return KernelSignature(
          "split", {"X"}, {"SectionsTensorList", "AxisTensor"}, {"Out"});
    } else {
      return KernelSignature(
          "split", {"X"}, {"SectionsTensorList", "axis"}, {"Out"});
    }
  }

  if (ctx.HasInput("AxisTensor")) {
    return KernelSignature("split", {"X"}, {"sections", "AxisTensor"}, {"Out"});
  } else {
    return KernelSignature("split", {"X"}, {"sections", "axis"}, {"Out"});
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(split, phi::SplitOpArgumentMapping);
