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

#pragma once

#include "paddle/phi/kernels/copy_kernel.h"

namespace phi {

template <typename T, typename Context>
void SizeKernel(const Context& ctx,
                const DenseTensor& input,
                DenseTensor* out) {
  auto place = ctx.GetPlace();
  auto out_data = ctx.template Alloc<int64_t>(out);
  auto cpu_place = phi::CPUPlace();
  if (place == cpu_place) {
    out_data[0] = input.numel();
  } else {
    DenseTensor cpu_tensor;
    cpu_tensor.Resize(out->dims());
    auto cpu_data = ctx.template HostAlloc<int64_t>(&cpu_tensor);
    cpu_data[0] = input.numel();
    phi::Copy(ctx, cpu_tensor, place, false, out);
  }
}

}  // namespace phi
