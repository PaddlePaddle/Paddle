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

KernelSignature ReduceSumOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.IsDenseTensorInput("X")) {
    bool reduce_all = paddle::any_cast<bool>(ctx.Attr("reduce_all"));
    // When ctx is InferShapeArgumentMappingContext, the reduce_all is used in
    // InferShape, so we must return the "sum_raw" KernelSignature.
    // And the InferMeta function(i.e. ReduceInferMetaBase) is accordance with
    // the "sum_raw" KernelSignature
    if (ctx.IsForInferShape() || reduce_all) {
      return KernelSignature("sum_raw",
                             {"X"},
                             {"dim", "keep_dim", "reduce_all", "out_dtype"},
                             {"Out"});
    }
    return KernelSignature(
        "sum", {"X"}, {"dim", "out_dtype", "keep_dim"}, {"Out"});
  }
  return KernelSignature("unregistered", {}, {}, {});
}

KernelSignature ReduceMeanOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.IsDenseTensorInput("X")) {
    bool reduce_all = paddle::any_cast<bool>(ctx.Attr("reduce_all"));
    // When ctx is InferShapeArgumentMappingContext, the reduce_all is used in
    // InferShape, so we must return the "mean_raw" KernelSignature.
    // And the InferMeta function(i.e. MeanRawInferMeta) is accordance with the
    // "mean_raw" KernelSignature
    if (ctx.IsForInferShape() || reduce_all) {
      return KernelSignature(
          "mean_raw", {"X"}, {"dim", "keep_dim", "reduce_all"}, {"Out"});
    }
    return KernelSignature("mean", {"X"}, {"dim", "keep_dim"}, {"Out"});
  }
  return KernelSignature("unregistered", {}, {}, {});
}

KernelSignature ReduceProdOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "reduce_prod", {"X"}, {"dim", "keep_dim", "reduce_all"}, {"Out"});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(reduce_sum, sum);
PD_REGISTER_BASE_KERNEL_NAME(reduce_mean, mean);

PD_REGISTER_ARG_MAPPING_FN(reduce_sum, phi::ReduceSumOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(reduce_mean, phi::ReduceMeanOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(reduce_prod, phi::ReduceProdOpArgumentMapping);
