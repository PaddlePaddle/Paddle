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

KernelSignature FillConstantBatchSizeLikeOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  const auto& str_value = paddle::any_cast<std::string>(ctx.Attr("str_value"));
  if (str_value.empty()) {
    return KernelSignature(
        "full_batch_size_like",
        {"Input"},
        {"shape", "value", "dtype", "input_dim_idx", "output_dim_idx"},
        {"Out"});
  } else {
    return KernelSignature(
        "full_batch_size_like",
        {"Input"},
        {"shape", "str_value", "dtype", "input_dim_idx", "output_dim_idx"},
        {"Out"});
  }
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(fill_constant_batch_size_like,
                             full_batch_size_like);

PD_REGISTER_ARG_MAPPING_FN(fill_constant_batch_size_like,
                           phi::FillConstantBatchSizeLikeOpArgumentMapping);
