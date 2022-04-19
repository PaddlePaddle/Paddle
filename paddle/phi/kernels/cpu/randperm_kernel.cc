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

#include "paddle/phi/kernels/randperm_kernel.h"

#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void RandpermRawKernel(
    const Context& dev_ctx, int n, DataType dtype, int seed, DenseTensor* out) {
  T* out_data = dev_ctx.template Alloc<T>(out);

  std::shared_ptr<std::mt19937_64> engine;
  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    engine = dev_ctx.GetGenerator()->GetCPUEngine();
  }

  for (int i = 0; i < n; ++i) {
    out_data[i] = static_cast<T>(i);
  }
  std::shuffle(out_data, out_data + n, *engine);
}

template <typename T, typename Context>
void RandpermKernel(const Context& dev_ctx,
                    int n,
                    DataType dtype,
                    DenseTensor* out) {
  RandpermRawKernel<T>(dev_ctx, n, dtype, 0, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(randperm_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::RandpermRawKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(randperm,
                   CPU,
                   ALL_LAYOUT,
                   phi::RandpermKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
