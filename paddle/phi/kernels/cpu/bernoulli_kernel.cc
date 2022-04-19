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

#include "paddle/phi/kernels/bernoulli_kernel.h"
#include <random>
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
inline T BernoulliFunctor(T p, T rand) {
  PADDLE_ENFORCE_LE(
      p,
      1.0,
      phi::errors::OutOfRange("The probability should be <= 1, but got %f", p));
  PADDLE_ENFORCE_GE(
      p,
      0.0,
      phi::errors::OutOfRange("The probability should be >= 0, but got %f", p));
  return static_cast<T>(rand < p);
}

template <typename T, typename Context>
void BernoulliKernel(const Context& ctx,
                     const DenseTensor& x,
                     DenseTensor* out) {
  auto numel = x.numel();
  auto* x_data = x.data<T>();
  T* out_data = ctx.template Alloc<T>(out);

  std::uniform_real_distribution<T> dist(0.0, 1.0);
  auto gen_ptr = ctx.GetGenerator();
  auto engine = gen_ptr->GetCPUEngine();

  for (int64_t i = 0; i < numel; ++i) {
    out_data[i] = BernoulliFunctor(x_data[i], dist(*engine));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    bernoulli, CPU, ALL_LAYOUT, phi::BernoulliKernel, float, double) {}
