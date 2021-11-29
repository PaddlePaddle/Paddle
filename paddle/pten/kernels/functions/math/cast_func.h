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

#pragma once

#include "paddle/fluid/platform/aligned_vector.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/transform.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/fluid/platform/gpu_launch_config.h"
#endif
#include "paddle/pten/core/dense_tensor.h"

namespace pten {
namespace math {

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename InT, typename OutT, int VecSize>
__global__ void VecCastCUDAKernel(const InT* in, const int64_t N, OutT* out) {
  using LoadT = paddle::platform::AlignedVector<InT, VecSize>;
  using StoreT = paddle::platform::AlignedVector<OutT, VecSize>;

  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int64_t i = idx * VecSize; i < N;
       i += blockDim.x * gridDim.x * VecSize) {
    LoadT in_val;
    paddle::platform::Load<InT, VecSize>(&in[i], &in_val);

    StoreT out_val;
#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      out_val[j] = static_cast<OutT>(in_val[j]);
    }

    paddle::platform::Store<OutT, VecSize>(out_val, &out[i]);
  }
}

template <typename InT, typename OutT>
__global__ void CastCUDAKernel(const InT* in, const int64_t N, OutT* out) {
  CUDA_KERNEL_LOOP(index, N) { out[index] = static_cast<OutT>(in[index]); }
}
#endif

template <typename InT, typename OutT>
struct CastOpTransformFunctor {
  HOSTDEVICE OutT operator()(InT in) const { return static_cast<OutT>(in); }
};

template <typename DeviceContext, typename InT, typename OutT>
void CastWithRawPtr(const DeviceContext& dev_ctx,
                    const InT* x,
                    OutT* out,
                    int64_t size) {
#if defined(__NVCC__) || defined(__HIPCC__)
  // NOTE: if constexpr would be better here, but it requires C++ 17
  if (std::is_same<DeviceContext, paddle::platform::CUDADeviceContext>::value) {
    const auto& cuda_dev_ctx =
        static_cast<const paddle::platform::CUDADeviceContext&>(dev_ctx);
    const auto& config =
        paddle::platform::GetGpuLaunchConfig1D(cuda_dev_ctx, size);
    auto vec_size = paddle::platform::GetVectorizedSize<OutT>(out);
    if (!std::is_same<InT, OutT>::value && vec_size == 4 && size % 4 == 0) {
      VecCastCUDAKernel<InT, OutT, 4><<<config.block_per_grid,
                                        config.thread_per_block,
                                        0,
                                        cuda_dev_ctx.stream()>>>(x, size, out);
    } else {
      CastCUDAKernel<InT, OutT><<<config.block_per_grid,
                                  config.thread_per_block,
                                  0,
                                  cuda_dev_ctx.stream()>>>(x, size, out);
    }
    return;
  }
#endif
  paddle::platform::Transform<DeviceContext> trans;
  trans(dev_ctx, x, x + size, out, CastOpTransformFunctor<InT, OutT>());
}

template <typename DeviceContext, typename InT, typename OutT>
void CastKernelImpl(const DeviceContext& dev_ctx,
                    const DenseTensor& x,
                    DenseTensor* out) {
  const auto* in_begin = x.data<InT>();
  auto* out_begin = out->mutable_data<OutT>();
  CastWithRawPtr<DeviceContext, InT, OutT>(
      dev_ctx, in_begin, out_begin, x.numel());
}

}  // namespace math

}  // namespace pten
