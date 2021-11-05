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
#include "paddle/fluid/operators/mean_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void MeanRunKernel(const T* in_data, T* out_data, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  T data = in_data[0];
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    out_data[idx] = data / (static_cast<T>(N));
  }
}

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
    mean, ops::MeanKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MeanKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MeanKernel<paddle::platform::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    mean_grad,
    ops::MeanCUDAGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MeanCUDAGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MeanCUDAGradKernel<paddle::platform::CUDADeviceContext,
                            plat::float16>);
