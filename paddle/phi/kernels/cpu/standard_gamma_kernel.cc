// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/standard_gamma_kernel.h"

#include <random>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void StandardGammaKernel(const Context& ctx,
                         const DenseTensor& x,
                         DenseTensor* out) {
  const T* x_data = x.data<T>();
  T* out_data = ctx.template Alloc<T>(out);
  int64_t size = x.numel();

  auto gen = ctx.GetGenerator();
  auto engine = gen->GetCPUEngine();

  for (int64_t i = 0; i < size; ++i) {
    std::gamma_distribution<> dist(x_data[i], 1.);
    out_data[i] = static_cast<T>(dist(*engine));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    standard_gamma, CPU, ALL_LAYOUT, phi::StandardGammaKernel, float, double) {}
