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

KernelSignature ReduceSumOpArgumentMapping(const ArgumentMappingContext& ctx) {
  bool reduce_all = paddle::any_cast<bool>(ctx.Attr("reduce_all"));
  if (ctx.IsDenseTensorInput("X")) {
    if (!reduce_all) {
      return KernelSignature(
          "sum", {"X"}, {"dim", "out_dtype", "keep_dim"}, {"Out"});
    }
    return KernelSignature("sum_raw",
                           {"X"},
                           {"dim", "keep_dim", "reduce_all", "out_dtype"},
                           {"Out"});
  }
  return KernelSignature("unregistered", {}, {}, {});
}

KernelSignature ReduceMeanOpArgumentMapping(const ArgumentMappingContext& ctx) {
  bool reduce_all = paddle::any_cast<bool>(ctx.Attr("reduce_all"));
  if (ctx.IsDenseTensorInput("X")) {
    if (!reduce_all) {
      return KernelSignature("mean", {"X"}, {"dim", "keep_dim"}, {"Out"});
    }
    return KernelSignature(
        "mean_raw", {"X"}, {"dim", "keep_dim", "reduce_all"}, {"Out"});
  }
  return KernelSignature("unregistered", {}, {}, {});
}

}  // namespace pten

PT_REGISTER_BASE_KERNEL_NAME(reduce_sum, sum);
PT_REGISTER_BASE_KERNEL_NAME(reduce_mean, mean);

PT_REGISTER_ARG_MAPPING_FN(reduce_sum, pten::ReduceSumOpArgumentMapping);
PT_REGISTER_ARG_MAPPING_FN(reduce_mean, pten::ReduceMeanOpArgumentMapping);
