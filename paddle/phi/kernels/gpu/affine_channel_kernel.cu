// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif

#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/affine_channel_kernel.h"

namespace phi {

template <typename T, phi::DataLayout layout, bool HasBias>
__global__ static inline void KeAffineChannelCUDA(const T* x,
                                                  const T* scale,
                                                  const T* bias,
                                                  const int C,
                                                  const int64_t HxW,
                                                  const int64_t num,
                                                  T* y) {
  int64_t gid =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
      static_cast<int64_t>(threadIdx.x);
  int stride = blockDim.x * gridDim.x;
  for (int64_t i = gid; i < num; i += stride) {
    const int c = layout == phi::DataLayout::NCHW ? i / HxW % C : i % C;
    if (HasBias) {
      y[i] = scale[c] * x[i] + bias[c];
    } else {
      y[i] = scale[c] * x[i];
    }
  }
}

template <typename T, typename Context>
void AffineChannelCUDAKernel(const Context& dev_ctx,
                             const DenseTensor& x_in,
                             const DenseTensor& scale_in,
                             const DenseTensor& bias_in,
                             const std::string& data_layout,
                             DenseTensor* out) {
  auto* x = &x_in;
  auto* scale = &scale_in;
  auto* bias = &bias_in;

  auto* y = out;
  dev_ctx.template Alloc<T>(y);

  const phi::DataLayout layout = common::StringToDataLayout(data_layout);

  auto dims = x->dims();
  const int64_t num = x->numel();
  int64_t N = dims[0];
  // TODO(large-tensor): downstream functors may still use int; guard until
  // upgraded.

  int C = layout == phi::DataLayout::NCHW ? dims[1] : dims[dims.size() - 1];
  int64_t HxW = num / N / C;

  const T* x_d = x->data<T>();
  const T* scale_d = scale->data<T>();
  const T* bias_d = bias->data<T>();
  T* y_d = y->data<T>();

#ifdef PADDLE_WITH_HIP
  int block = 256;
#else
  int block = 1024;
#endif  // PADDLE_WITH_HIP
  int grid = (num + block - 1) / block;

  int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  grid = std::min(std::max(max_threads / block, 1), grid);
  if (layout == phi::DataLayout::NCHW) {
    KeAffineChannelCUDA<T, phi::DataLayout::NCHW, true>
        <<<grid, block, 0, dev_ctx.stream()>>>(
            x_d, scale_d, bias_d, C, HxW, num, y_d);
  } else {
    KeAffineChannelCUDA<T, phi::DataLayout::NHWC, true>
        <<<grid, block, 0, dev_ctx.stream()>>>(
            x_d, scale_d, bias_d, C, HxW, num, y_d);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(affine_channel,
                   GPU,
                   ALL_LAYOUT,
                   phi::AffineChannelCUDAKernel,
                   float,
                   double) {}
