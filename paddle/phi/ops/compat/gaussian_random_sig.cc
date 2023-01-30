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

KernelSignature GaussianRandomOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  const auto& shape = paddle::any_cast<std::vector<int64_t>>(ctx.Attr("shape"));
  if (ctx.InputSize("ShapeTensorList") > 0) {
    // Infer output shape by Attr("shape") in CompileTime if it is specified.
    if (!ctx.IsRuntime() && !shape.empty()) {
<<<<<<< HEAD
      return KernelSignature(
          "gaussian", {}, {"shape", "mean", "std", "seed", "dtype"}, {"Out"});
    } else {
      return KernelSignature(
          "gaussian",
=======
      return KernelSignature("gaussian_random",
                             {},
                             {"shape", "mean", "std", "seed", "dtype"},
                             {"Out"});
    } else {
      return KernelSignature(
          "gaussian_random",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
          {},
          {"ShapeTensorList", "mean", "std", "seed", "dtype"},
          {"Out"});
    }
  }

  if (ctx.HasInput("ShapeTensor") && shape.empty()) {
<<<<<<< HEAD
    return KernelSignature("gaussian",
=======
    return KernelSignature("gaussian_random",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                           {},
                           {"ShapeTensor", "mean", "std", "seed", "dtype"},
                           {"Out"});
  }

<<<<<<< HEAD
  return KernelSignature(
      "gaussian", {}, {"shape", "mean", "std", "seed", "dtype"}, {"Out"});
=======
  return KernelSignature("gaussian_random",
                         {},
                         {"shape", "mean", "std", "seed", "dtype"},
                         {"Out"});
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}

}  // namespace phi

<<<<<<< HEAD
PD_REGISTER_BASE_KERNEL_NAME(gaussian_random, gaussian);

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
PD_REGISTER_ARG_MAPPING_FN(gaussian_random,
                           phi::GaussianRandomOpArgumentMapping);
