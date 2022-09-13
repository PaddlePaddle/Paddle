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

#include "paddle/phi/kernels/pad_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/onednn/pad_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void PadKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int>& paddings,
               const Scalar& pad_value,
               DenseTensor* out) {
  std::vector<int64_t> copied_paddings(paddings.begin(), paddings.end());

  std::swap(copied_paddings[0], copied_paddings[2]);
  std::swap(copied_paddings[1], copied_paddings[3]);
  PadOpKernel<T, Context>(
      dev_ctx, x, copied_paddings, pad_value.to<float>(), out);
}
}  // namespace phi

PD_REGISTER_KERNEL(pad, OneDNN, ALL_LAYOUT, phi::PadKernel, float) {}
