/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/uniform_random_inplace_kernel.h"

#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void UniformRandomInplaceKernel(const Context& ctx,
                                const DenseTensor& x,
                                float min,
                                float max,
                                int seed,
                                int diag_num,
                                int diag_step,
                                float diag_val,
                                DenseTensor* out) {
  T* data = ctx.template Alloc<T>(out);
  int64_t size = out->numel();
  std::uniform_real_distribution<T> dist(static_cast<T>(min),
                                         static_cast<T>(max));
  std::shared_ptr<std::mt19937_64> engine;
  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    engine = ctx.GetGenerator()->GetCPUEngine();
  }
  for (int64_t i = 0; i < size; ++i) {
    data[i] = dist(*engine);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(uniform_random_inplace,
                   CPU,
                   ALL_LAYOUT,
                   phi::UniformRandomInplaceKernel,
                   float,
                   double) {}
