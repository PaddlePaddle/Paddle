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

#include "paddle/phi/kernels/nll_loss_kernel.h"

namespace phi {
template <typename T, typename Context>
void NllLossKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   const DenseTensor& label,
                   paddle::optional<const DenseTensor&> weight,
                   int64_t ignore_index,
                   const std::string& reduction,
                   DenseTensor* out) {
  DenseTensor total_weight;
  total_weight.set_meta(
      DenseTensorMeta(paddle::experimental::CppTypeToDataType<T>::Type(), {1}));
  dev_ctx.template Alloc<T>(total_weight);
  NllLossRawKernel(dev_ctx,
                   input,
                   label,
                   weight,
                   ignore_index,
                   reduction,
                   out,
                   &total_weight);
}
}  // namespace phi

// TODO(xiongkun): add the non-raw kernel register here.
