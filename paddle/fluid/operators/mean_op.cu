/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/kernel_primitives/functor_primitives.h"
#include "paddle/fluid/operators/mean_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void MeanRunKernel(const T* in_data, T* out_data, int N) {
  using MT = typename details::MPTypeTrait<T>::Type;
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  auto data = static_cast<MT>(in_data[0]);
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    out_data[idx] = static_cast<T>(data / (static_cast<MT>(N)));
  }
}

template <typename DeviceContext, typename T>
class MeanCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");

    const T* in_data = input->data<T>();
    T* out_data = output->mutable_data<T>(context.GetPlace());
    auto numel = input->numel();
    auto rank = input->dims().size();
    auto place = context.GetPlace();
    auto stream = context.cuda_device_context().stream();

    if (rank == 0) {  // scalar
      auto gpu_place = place;
      memory::Copy(gpu_place, out_data, gpu_place, in_data, numel * sizeof(T),
                   stream);
      return;
    }

    using MT = typename details::MPTypeTrait<T>::Type;
    using Div = kernel_primitives::DivideFunctor<T, MT>;
    std::vector<int> reduce_dims;
    reduce_dims.reserve(rank);
    for (decltype(rank) i = 0; i < rank; ++i) {
      reduce_dims.push_back(i);
    }
    TensorReduceFunctorImpl<T, T, kernel_primitives::AddFunctor, Div>(
        *input, output, Div(numel), reduce_dims, stream);
  }
};

template <typename DeviceContext, typename T>
class MeanCUDAGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto OG = context.Input<Tensor>(framework::GradVarName("Out"));
    PADDLE_ENFORCE_EQ(OG->numel(), 1,
                      platform::errors::InvalidArgument(
                          "Mean Gradient Input Tensor len should be 1. But "
                          "received Out@Grad's elements num is %d.",
                          OG->numel()));
    auto IG = context.Output<Tensor>(framework::GradVarName("X"));
    IG->mutable_data<T>(context.GetPlace());

    auto in_data = OG->data<T>();
    auto size_prob = IG->numel();
    auto out_data = IG->data<T>();
    int threads = 512;
    int grid = (size_prob + threads - 1) / threads;
    auto stream = context.cuda_device_context().stream();
    MeanRunKernel<T><<<grid, threads, 0, stream>>>(in_data, out_data,
                                                   size_prob);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    mean, ops::MeanCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MeanCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MeanCUDAKernel<paddle::platform::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    mean_grad,
    ops::MeanCUDAGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MeanCUDAGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MeanCUDAGradKernel<paddle::platform::CUDADeviceContext,
                            plat::float16>);
