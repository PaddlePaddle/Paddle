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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/elementwise.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"
#include "paddle/phi/kernels/impl/dirichlet_kernel_impl.h"

namespace phi {

template <typename T, typename UniformSamplerT, typename NormalSamplerT>
struct GammaCPUFunctor {
  GammaCPUFunctor(const T* alpha,
                  T* gamma,
                  BaseSampler<T, UniformSamplerT> uniform,
                  BaseSampler<T, NormalSamplerT> normal)
      : alpha_(alpha), gamma_(gamma), uniform_(uniform), normal_(normal) {}

  HOST void operator()(int64_t index) {
    auto sample = sample_gamma<T, T, UniformSamplerT, NormalSamplerT>(
        alpha_[index], uniform_, normal_);
    gamma_[index] = std::max(std::numeric_limits<T>::min(), sample);
  }

  const T* alpha_;
  T* gamma_;
  BaseSampler<T, UniformSamplerT> uniform_;
  BaseSampler<T, NormalSamplerT> normal_;
};

template <typename T>
struct DirichletSampler<CPUContext, T> {
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

    // sample from K gamma distributions, where K=alpha.numel()
    DenseTensor gamma_samples;
    gamma_samples.Resize(alpha.dims());
    dev_ctx.template Alloc<T>(&gamma_samples);

    GammaCPUFunctor<T, decltype(uniform), decltype(normal)> gamma_functor(
        alpha.data<T>(),
        gamma_samples.data<T>(),
        standard_uniform,
        standard_normal);
    funcs::ForRange<CPUContext> for_range(dev_ctx, alpha.numel());
    for_range(gamma_functor);

    // normalize them into a simplex, along the last axis
    DenseTensor gamma_sum;
    auto new_shape = gamma_samples.dims();
    new_shape[new_shape.size() - 1] = 1;
    gamma_sum.Resize(new_shape);
    dev_ctx.template Alloc<T>(&gamma_sum);

    funcs::ReduceKernelImpl<CPUContext, T, T, funcs::SumFunctor>(
        dev_ctx,
        gamma_samples,
        &gamma_sum,
        {new_shape.size() - 1},
        true,
        false);

    funcs::ElementwiseCompute<funcs::DivideFunctor<T>, T, T>(
        dev_ctx, gamma_samples, gamma_sum, -1, funcs::DivideFunctor<T>(), out);
  }
};

}  // namespace phi

PD_REGISTER_KERNEL(
    dirichlet, CPU, ALL_LAYOUT, phi::Dirichletkernel, float, double) {}
