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

#include "paddle/pten/kernels/cast_kernel.h"

#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/kernel_registry.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/aligned_vector.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/device/gpu/gpu_helper.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/float16.h"

namespace pten {

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

template <typename InT, typename OutT>
void CastCUDAKernelImpl(const GPUContext& dev_ctx,
                        const DenseTensor& x,
                        DenseTensor* out) {
  auto* in_data = x.data<InT>();
  auto size = x.numel();
  auto* out_data = out->mutable_data<OutT>();

  paddle::platform::GpuLaunchConfig config =
      paddle::platform::GetGpuLaunchConfig1D(dev_ctx, size);
  int vec_size = paddle::platform::GetVectorizedSize<OutT>(out_data);
  if (!std::is_same<InT, OutT>::value && vec_size == 4 && size % 4 == 0) {
    VecCastCUDAKernel<InT, OutT, 4><<<config.block_per_grid,
                                      config.thread_per_block,
                                      0,
                                      dev_ctx.stream()>>>(
        in_data, size, out_data);
  } else {
    CastCUDAKernel<InT, OutT><<<config.block_per_grid,
                                config.thread_per_block,
                                0,
                                dev_ctx.stream()>>>(in_data, size, out_data);
  }
}

template <typename T, typename ContextT>
void Cast(const ContextT& dev_ctx,
          const DenseTensor& x,
          DataType out_dtype,
          DenseTensor* out) {
  PD_VISIT_ALL_TYPES(out_dtype, "CastCUDAKernelImpl", ([&] {
                       CastCUDAKernelImpl<T, data_t>(dev_ctx, x, out);
                     }));
}

}  // namespace pten

#define PTEN_REGISTER_CAST_CUDA_BASE_TYPE(op_name, ...)     \
  PT_REGISTER_CTX_KERNEL(cast,                              \
                         GPU,                               \
                         ALL_LAYOUT,                        \
                         pten::Cast,                        \
                         float,                             \
                         double,                            \
                         int,                               \
                         int64_t,                           \
                         int16_t,                           \
                         bool,                              \
                         uint8_t,                           \
                         paddle::platform::float16,         \
                         paddle::platform::complex<float>,  \
                         paddle::platform::complex<double>, \
                         ##__VA_ARGS__) {                   \
    kernel->OutputAt(0).SetDataType(                        \
        paddle::experimental::DataType::UNDEFINED);         \
  }

#if !defined(PADDLE_WITH_HIP)
PTEN_REGISTER_CAST_CUDA_BASE_TYPE(cast, paddle::platform::bfloat16)
#else
PTEN_REGISTER_CAST_CUDA_BASE_TYPE(cast)
#endif
