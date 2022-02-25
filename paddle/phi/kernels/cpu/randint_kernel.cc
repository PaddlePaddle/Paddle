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

#include "paddle/phi/kernels/randint_kernel.h"

#include <random>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void RandintRawKernel(const Context& ctx,
                      int low,
                      int high,
                      const ScalarArray& shape,
                      DataType dtype,
                      int seed,
                      DenseTensor* out) {
  out->ResizeAndAllocate(phi::make_ddim(shape.GetData()));
  auto size = out->numel();
  std::shared_ptr<std::mt19937_64> engine;
  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    engine = ctx.GetGenerator()->GetCPUEngine();
  }
  std::uniform_int_distribution<T> dist(low, high - 1);
  auto data = out->data<T>();
  for (int64_t i = 0; i < size; ++i) {
    data[i] = dist(*engine);
  }
}

template <typename T, typename Context>
void RandintKernel(const Context& ctx,
                   int low,
                   int high,
                   const ScalarArray& shape,
                   DataType dtype,
                   DenseTensor* out) {
  RandintRawKernel<T>(ctx, low, high, shape, dtype, 0, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    randint_raw, CPU, ALL_LAYOUT, phi::RandintRawKernel, int, int64_t) {}
PD_REGISTER_KERNEL(randint, CPU, ALL_LAYOUT, phi::RandintKernel, int, int64_t) {
}
