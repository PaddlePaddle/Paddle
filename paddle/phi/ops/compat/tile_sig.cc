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

KernelSignature TileOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("RepeatTimes")) {
    return KernelSignature("tile", {"X"}, {"RepeatTimes"}, {"Out"});
  } else if (ctx.InputSize("repeat_times_tensor") > 0) {
    const auto& repeat_times =
        paddle::any_cast<std::vector<int>>(ctx.Attr("repeat_times"));
    if (!ctx.IsRuntime() && !repeat_times.empty()) {
      return KernelSignature("tile", {"X"}, {"repeat_times"}, {"Out"});
    }
    return KernelSignature("tile", {"X"}, {"repeat_times_tensor"}, {"Out"});
  } else {
    return KernelSignature("tile", {"X"}, {"repeat_times"}, {"Out"});
  }
}

KernelSignature TileGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("RepeatTimes")) {
    return KernelSignature(
        "tile_grad", {"X", "Out@GRAD"}, {"RepeatTimes"}, {"X@GRAD"});
  } else if (ctx.InputSize("repeat_times_tensor") > 0) {
    return KernelSignature(
        "tile_grad", {"X", "Out@GRAD"}, {"repeat_times_tensor"}, {"X@GRAD"});
  } else {
    return KernelSignature(
        "tile_grad", {"X", "Out@GRAD"}, {"repeat_times"}, {"X@GRAD"});
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(tile, phi::TileOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(tile_grad, phi::TileGradOpArgumentMapping);
