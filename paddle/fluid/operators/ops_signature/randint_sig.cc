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

KernelSignature RandintOpArgumentMapping(const ArgumentMappingContext& ctx) {
  int seed = paddle::any_cast<int>(ctx.Attr("seed"));
  if (seed) {
    if (ctx.InputSize("ShapeTensorList") > 0) {
      return KernelSignature(
          "randint_with_seed",
          {},
          {"low", "high", "ShapeTensorList", "seed", "dtype"},
          {"Out"});
    } else {
      const auto& shape =
          paddle::any_cast<std::vector<int64_t>>(ctx.Attr("shape"));
      if (ctx.HasInput("ShapeTensor") && shape.empty()) {
        return KernelSignature("randint_with_seed",
                               {},
                               {"low", "high", "ShapeTensor", "seed", "dtype"},
                               {"Out"});
      } else {
        return KernelSignature("randint_with_seed",
                               {},
                               {"low", "high", "shape", "seed", "dtype"},
                               {"Out"});
      }
    }
  } else {
    if (ctx.InputSize("ShapeTensorList") > 0) {
      return KernelSignature(
          "randint", {}, {"low", "high", "ShapeTensorList", "dtype"}, {"Out"});
    } else {
      const auto& shape =
          paddle::any_cast<std::vector<int64_t>>(ctx.Attr("shape"));
      if (ctx.HasInput("ShapeTensor") && shape.empty()) {
        return KernelSignature(
            "randint", {}, {"low", "high", "ShapeTensor", "dtype"}, {"Out"});
      } else {
        return KernelSignature(
            "randint", {}, {"low", "high", "shape", "dtype"}, {"Out"});
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(randint, phi::RandintOpArgumentMapping);
