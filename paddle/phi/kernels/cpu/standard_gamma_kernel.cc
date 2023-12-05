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

#include "paddle/phi/kernels/impl/standard_gamma_kernel_impl.h"

#include <random>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/dirichlet_util.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T>
struct GammaSampler<CPUContext, T> {
  void operator()(const CPUContext& dev_ctx,
                  const DenseTensor& alpha,
                  DenseTensor* out) {
    auto generator = dev_ctx.GetGenerator()->GetCPUEngine();

    auto uniform = [&generator]() -> T {
      std::uniform_real_distribution<T> u(0.0, 1.0);
      return u(*generator);
    };
    BaseSampler<T, decltype(uniform)> standard_uniform(uniform);

    auto normal = [&generator]() {
      std::normal_distribution<T> n(0.0, 1.0);
      return n(*generator);
    };
    BaseSampler<T, decltype(normal)> standard_normal(normal);

    GammaCPUFunctor<T, decltype(uniform), decltype(normal)> gamma_functor(
        alpha.data<T>(), out->data<T>(), standard_uniform, standard_normal);
    funcs::ForRange<CPUContext> for_range(dev_ctx, out->numel());
    for_range(gamma_functor);
  }
};

}  // namespace phi

PD_REGISTER_KERNEL(
    standard_gamma, CPU, ALL_LAYOUT, phi::StandardGammaKernel, float, double) {}
