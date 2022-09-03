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

#include "paddle/phi/kernels/truncated_gaussian_random_kernel.h"

#include <limits>
#include <random>
#include <vector>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/truncated_normal.h"

namespace phi {

template <typename T, typename Context>
void TruncatedGaussianRandomKernel(const Context& dev_ctx,
                                   const std::vector<int>& shape,
                                   float mean,
                                   float std,
                                   int seed,
                                   DataType dtype,
                                   DenseTensor* out) {
  auto tensor = out;

  T* data = dev_ctx.template Alloc<T>(tensor);

  std::uniform_real_distribution<T> dist(std::numeric_limits<float>::min(),
                                         1.0);
  TruncatedNormal<T> truncated_normal(mean, std);
  int64_t size = tensor->numel();

  std::shared_ptr<std::mt19937_64> engine;
  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    engine = dev_ctx.GetGenerator()->GetCPUEngine();
  }
  for (int64_t i = 0; i < size; ++i) {
    data[i] = truncated_normal(dist(*engine));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(truncated_gaussian_random,
                   CPU,
                   ALL_LAYOUT,
                   phi::TruncatedGaussianRandomKernel,
                   float) {}
