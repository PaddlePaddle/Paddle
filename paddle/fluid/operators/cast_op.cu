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
#include "paddle/fluid/platform/aligned_vector.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/gpu_launch_config.h"

namespace paddle {
namespace operators {

template <typename InT, typename OutT, int VecSize>
__global__ void VecCastCUDAKernel(const InT* in, const int64_t N, OutT* out) {
  using LoadT = platform::AlignedVector<InT, VecSize>;
  using StoreT = platform::AlignedVector<OutT, VecSize>;

  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int64_t i = idx * VecSize; i < N;
       i += blockDim.x * gridDim.x * VecSize) {
    LoadT in_val;
    platform::Load<InT, VecSize>(&in[i], &in_val);

    StoreT out_val;
#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      out_val[j] = static_cast<OutT>(in_val[j]);
    }

    platform::Store<OutT, VecSize>(out_val, &out[i]);
  }
}

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
    int vec_size = platform::GetVectorizedSize<OutT>(out);
    if (!std::is_same<InT, OutT>::value && vec_size == 4 && size % 4 == 0) {
      VecCastCUDAKernel<InT, OutT, 4><<<
          config.block_per_grid, config.thread_per_block, 0, ctx_.stream()>>>(
          in, size, out);
    } else {
      CastCUDAKernel<InT, OutT><<<config.block_per_grid,
                                  config.thread_per_block, 0, ctx_.stream()>>>(
          in, size, out);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

#define REGISTER_CAST_CUDA_BASE(op_name, ...)                            \
  REGISTER_OP_CUDA_KERNEL(                                               \
      op_name, ops::CastOpKernel<plat::CUDADeviceContext, float>,        \
      ops::CastOpKernel<plat::CUDADeviceContext, double>,                \
      ops::CastOpKernel<plat::CUDADeviceContext, int>,                   \
      ops::CastOpKernel<plat::CUDADeviceContext, int64_t>,               \
      ops::CastOpKernel<plat::CUDADeviceContext, int16_t>,               \
      ops::CastOpKernel<plat::CUDADeviceContext, bool>,                  \
      ops::CastOpKernel<plat::CUDADeviceContext, uint8_t>,               \
      ops::CastOpKernel<plat::CUDADeviceContext, plat::float16>,         \
      ops::CastOpKernel<plat::CUDADeviceContext, plat::complex<float>>,  \
      ops::CastOpKernel<plat::CUDADeviceContext, plat::complex<double>>, \
      ##__VA_ARGS__);

#if !defined(PADDLE_WITH_HIP)
REGISTER_CAST_CUDA_BASE(
    cast, ops::CastOpKernel<plat::CUDADeviceContext, plat::bfloat16>)
#else
REGISTER_CAST_CUDA_BASE(cast)
#endif
