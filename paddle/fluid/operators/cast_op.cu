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

#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/gpu_launch_config.h"

namespace paddle {
namespace operators {

template <typename InT, typename OutT>
__global__ void CastCUDAKernel(const InT* in, const int64_t N, OutT* out) {
  CUDA_KERNEL_LOOP(index, N) { out[index] = static_cast<OutT>(in[index]); }
}

template <typename InT>
struct CastOpFunctor<platform::CUDADeviceContext, InT> {
  const framework::Tensor* in_;
  framework::Tensor* out_;
  const platform::CUDADeviceContext& ctx_;
  CastOpFunctor(const framework::Tensor* in, framework::Tensor* out,
                const platform::CUDADeviceContext& ctx)
      : in_(in), out_(out), ctx_(ctx) {}

  template <typename OutT>
  void apply() const {
    auto* in = in_->data<InT>();
    auto size = in_->numel();
    auto* out = out_->mutable_data<OutT>(ctx_.GetPlace());
    platform::GpuLaunchConfig config =
        platform::GetGpuLaunchConfig1D(ctx_, size);
    CastCUDAKernel<InT, OutT><<<config.block_per_grid, config.thread_per_block,
                                0, ctx_.stream()>>>(in, size, out);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    cast, ops::CastOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CastOpKernel<paddle::platform::CUDADeviceContext, double>,
    ops::CastOpKernel<paddle::platform::CUDADeviceContext, int>,
    ops::CastOpKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::CastOpKernel<paddle::platform::CUDADeviceContext, bool>,
    ops::CastOpKernel<paddle::platform::CUDADeviceContext, uint8_t>,
    ops::CastOpKernel<paddle::platform::CUDADeviceContext,
                      paddle::platform::float16>,
    ops::CastOpKernel<paddle::platform::CUDADeviceContext,
                      paddle::platform::complex64>,
    ops::CastOpKernel<paddle::platform::CUDADeviceContext,
                      paddle::platform::complex128>);
