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

#ifdef __NVCC__
#include <curand_kernel.h>
#endif
#ifdef __HIPCC__
#include <hiprand_kernel.h>
#endif
#include "paddle/fluid/operators/poisson_op.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename T>
struct PoissonCudaFunctor {
 public:
  PoissonCudaFunctor(const T* in, T* out, unsigned int seed,
                     unsigned int offset)
      : in_(in), out_(out), seed_(seed), offset_(offset) {}

  __device__ void operator()(int64_t idx) {
#ifdef __NVCC__
    curandStatePhilox4_32_10_t state;
    curand_init(seed_, idx, offset_, &state);
    out_[idx] = static_cast<T>(curand_poisson(&state, in_[idx]));
#elif __HIPCC__
    hiprandStatePhilox4_32_10_t state;
    hiprand_init(seed_, idx, offset_, &state);
    out_[idx] = static_cast<T>(hiprand_poisson(&state, in_[idx]));
#endif
  }

 private:
  const T* in_;
  T* out_;
  const unsigned int seed_;
  const unsigned int offset_;
};

template <typename T>
class PoissonKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* x = ctx.Input<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");

    const T* x_data = x->data<T>();
    T* out_data = out->mutable_data<T>(ctx.GetPlace());
    auto size = x->numel();
    int64_t device_id = ctx.GetPlace().GetDeviceId();

    auto gen_cuda = framework::GetDefaultCUDAGenerator(device_id);
    auto seed_offset = gen_cuda->IncrementOffset(20);
    uint64_t seed = seed_offset.first;
    uint64_t offset = seed_offset.second;

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    platform::ForRange<platform::CUDADeviceContext> for_range(dev_ctx, size);

    PoissonCudaFunctor<T> functor(x_data, out_data, seed, offset);
    for_range(functor);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(poisson,
                        ops::PoissonKernel<plat::CUDADeviceContext, float>,
                        ops::PoissonKernel<plat::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    poisson_grad, ops::PoissonGradKernel<plat::CUDADeviceContext, float>,
    ops::PoissonGradKernel<plat::CUDADeviceContext, double>);
