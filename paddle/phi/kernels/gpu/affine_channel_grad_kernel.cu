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

namespace phi {

template <typename T, phi::DataLayout layout, bool HasBias>
__global__ static inline void KeAffineChannelCUDA(const T* x,
                                                  const T* scale,
                                                  const T* bias,
                                                  const int C,
                                                  const int HxW,
                                                  const int num,
                                                  T* y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = gid; i < num; i += stride) {
    const int c = layout == phi::DataLayout::kNCHW ? i / HxW % C : i % C;
    if (HasBias) {
      y[i] = scale[c] * x[i] + bias[c];
    } else {
      y[i] = scale[c] * x[i];
    }
  }
}

template <typename T, int BlockDim, phi::DataLayout layout>
__global__ void AffineChannelScaleBiasGradientCUDAKernel(const T* dy,
                                                         const T* x,
                                                         const int N,
                                                         const int C,
                                                         const int HxW,
                                                         T* dscale,
                                                         T* dbias) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  typedef cub::BlockReduce<double, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ds_storage;
  __shared__ typename BlockReduce::TempStorage db_storage;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T ds_sum = 0;
    T db_sum = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == phi::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      ds_sum += dy[index] * x[index];
      db_sum += dy[index];
    }
    __syncthreads();
    auto ds_out =
        BlockReduce(ds_storage).Reduce(static_cast<double>(ds_sum), cub::Sum());
    auto db_out =
        BlockReduce(db_storage).Reduce(static_cast<double>(db_sum), cub::Sum());
    __syncthreads();
    if (threadIdx.x == 0) {
      dscale[i] = ds_out;
      dbias[i] = db_out;
    }
  }
}

template <typename T, typename Context>
void AffineChannelGradCUDAKernel(const Context& dev_ctx,
                                 const DenseTensor& x_in,
                                 const DenseTensor& scale_in,
                                 const DenseTensor& bias_in,
                                 const DenseTensor& out_grad,
                                 const std::string& data_layout,
                                 DenseTensor* x_grad,
                                 DenseTensor* scale_grad,
                                 DenseTensor* bias_grad) {
  auto* x = &x_in;
  auto* scale = &scale_in;
  auto* bias = &bias_in;
  auto* dy = &out_grad;

  auto* dx = x_grad;
  auto* dscale = scale_grad;
  auto* dbias = bias_grad;

  const phi::DataLayout layout = common::StringToDataLayout(data_layout);

  auto dims = dy->dims();
  const int num = dy->numel();
  int N = dims[0];
  int C = layout == phi::DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
  int HxW = num / N / C;

  const T* dy_d = dy->data<T>();
  const T* s_d = scale->data<T>();

  T* dx_d = dx ? dev_ctx.template Alloc<T>(dx) : nullptr;
  T* ds_d = dscale ? dev_ctx.template Alloc<T>(dscale) : nullptr;
  T* db_d = dbias ? dev_ctx.template Alloc<T>(dbias) : nullptr;

#ifdef PADDLE_WITH_HIP
  const int block = 256;
#else
  const int block = 1024;
#endif  // PADDLE_WITH_HIP
  int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  const int max_blocks = std::max(max_threads / block, 1);
  int grid1 = (num + block - 1) / block;
  int grid2 = std::min(C, max_blocks);
  if (layout == phi::DataLayout::kNCHW) {
    if (dscale && dbias) {
      const T* x_d = x->data<T>();
      AffineChannelScaleBiasGradientCUDAKernel<T, block, phi::DataLayout::kNCHW>
          <<<grid2, block, 0, dev_ctx.stream()>>>(
              dy_d, x_d, N, C, HxW, ds_d, db_d);
    }
    if (dx) {
      KeAffineChannelCUDA<T, phi::DataLayout::kNCHW, false>
          <<<grid1, block, 0, dev_ctx.stream()>>>(
              dy_d, s_d, nullptr, C, HxW, num, dx_d);
    }
  } else {
    if (dscale && dbias) {
      const T* x_d = x->data<T>();
      AffineChannelScaleBiasGradientCUDAKernel<T, block, phi::DataLayout::kNHWC>
          <<<grid2, block, 0, dev_ctx.stream()>>>(
              dy_d, x_d, N, C, HxW, ds_d, db_d);
    }

    if (dx) {
      KeAffineChannelCUDA<T, phi::DataLayout::kNHWC, false>
          <<<grid1, block, 0, dev_ctx.stream()>>>(
              dy_d, s_d, nullptr, C, HxW, num, dx_d);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(affine_channel_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::AffineChannelGradCUDAKernel,
                   float,
                   double) {}
