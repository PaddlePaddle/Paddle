// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/operators/dirichlet_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_sum_op.h"
#include "paddle/fluid/platform/for_range.h"

#ifdef PADDLE_WITH_CUDA
#include <curand_kernel.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hiprand_kernel.h>
#endif

namespace paddle {
namespace operators {
template <typename T>
struct GammaCUDAFunctor {
  GammaCUDAFunctor(const T* alpha, T* gamma, uint64_t seed, uint64_t offset)
      : alpha_(alpha), gamma_(gamma), seed_(seed), offset_(offset) {}

  DEVICE void operator()(int64_t index) {
    // curand initialization
    curandStatePhilox4_32_10_t state;
    curand_init(/*seed=*/seed_, /*subsequence=*/index, /*offset=*/offset_,
                &state);

    // sample
    auto uniform_lambda = [&state]() { return curand_uniform(&state); };
    BaseSampler<T, decltype(uniform_lambda)> standard_uniform(uniform_lambda);
    auto normal_lambda = [&state]() { return curand_normal(&state); };
    BaseSampler<T, decltype(normal_lambda)> standard_normal(normal_lambda);

    auto sample =
        sample_gamma<T, T, decltype(uniform_lambda), decltype(normal_lambda)>(
            alpha_[index], standard_uniform, standard_normal);
    gamma_[index] = std::max(std::numeric_limits<T>::min(), sample);
  }

  const T* alpha_;
  T* gamma_;
  const uint64_t seed_;
  const uint64_t offset_;
};

template <typename T>
struct DirichletSampler<platform::CUDADeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor* alpha, framework::Tensor* out) {
    auto& dev_ctx = ctx.device_context<platform::CUDADeviceContext>();

    // init state, seed & offset for all threads
    auto p_gen = framework::GetDefaultCUDAGenerator();
    auto seed_and_offset = p_gen->IncrementOffset(10);  // hard-coded offset
    auto seed = seed_and_offset.first;
    auto offset = seed_and_offset.second;

    // sample from K gamma distributions, where K=alpha.numel()
    framework::Tensor gamma_samples;
    gamma_samples.mutable_data<T>(alpha->dims(), dev_ctx.GetPlace());
    GammaCUDAFunctor<T> gamma_functor(alpha->data<T>(), gamma_samples.data<T>(),
                                      seed, offset);
    platform::ForRange<platform::CUDADeviceContext> for_range(dev_ctx,
                                                              out->numel());
    for_range(gamma_functor);

    // normalize them into a simplex, along the last axis
    framework::Tensor gamma_sum;
    auto new_shape = gamma_samples.dims();
    new_shape[new_shape.size() - 1] = 1;
    gamma_sum.mutable_data<T>(new_shape, dev_ctx.GetPlace());

    ReduceKernelFunctor<platform::CPUDeviceContext, T, SumFunctor>(
        &gamma_samples, &gamma_sum, {new_shape.size() - 1}, true, false, ctx)
        .template apply<T>();
    ElementwiseComputeEx<DivFunctor<T>, platform::CPUDeviceContext, T, T>(
        ctx, &gamma_samples, &gamma_sum, -1, DivFunctor<T>(), out);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    dirichlet, ops::DirichletKernel<paddle::platform::CUDADeviceContext, float>,
    ops::DirichletKernel<paddle::platform::CUDADeviceContext, double>);
