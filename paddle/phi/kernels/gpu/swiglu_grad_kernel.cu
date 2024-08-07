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

#include "paddle/phi/kernels/swiglu_grad_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"

#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"

namespace phi {

template <typename T, int VecSize, bool IsCombine, bool HasDX, bool HasDY>
__global__ void SwiGLUGradCUDAKernel(const T *__restrict__ x,
                                     const T *__restrict__ y,
                                     const T *__restrict__ dz,
                                     T *__restrict__ dx,
                                     T *__restrict__ dy,
                                     int64_t m,
                                     int64_t n) {
  funcs::SwiGLUGradFunctor<T, HasDX, HasDY> functor;
  if constexpr (IsCombine) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t n_vec_piece = n / VecSize;
    int64_t valid_num = m * n_vec_piece;
    while (idx < valid_num) {
      int64_t row_offset = idx / n_vec_piece * n;
      int64_t col_offset = idx % n_vec_piece * VecSize;

      int64_t dz_offset = row_offset + col_offset;
      int64_t x_offset = dz_offset + row_offset;
      phi::AlignedVector<T, VecSize> x_vec;
      phi::AlignedVector<T, VecSize> y_vec;
      phi::AlignedVector<T, VecSize> dz_vec;

      phi::Load<T, VecSize>(x + x_offset, &x_vec);
      phi::Load<T, VecSize>(y + x_offset, &y_vec);
      phi::Load<T, VecSize>(dz + dz_offset, &dz_vec);

#pragma unroll
      for (int i = 0; i < VecSize; ++i) {
        functor(x_vec[i], y_vec[i], dz_vec[i], &x_vec[i], &y_vec[i]);
      }

      phi::Store<T, VecSize>(x_vec, dx + x_offset);
      phi::Store<T, VecSize>(y_vec, dy + x_offset);
      idx += stride;
    }
  } else {
    int64_t idx =
        (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * VecSize;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x * VecSize;
    int64_t numel = m * n;
    int64_t limit = numel - VecSize;

    while (idx <= limit) {
      phi::AlignedVector<T, VecSize> x_vec;
      phi::AlignedVector<T, VecSize> y_vec;
      phi::AlignedVector<T, VecSize> dz_vec;

      phi::Load<T, VecSize>(x + idx, &x_vec);
      if constexpr (HasDX) {
        phi::Load<T, VecSize>(y + idx, &y_vec);
      }
      phi::Load<T, VecSize>(dz + idx, &dz_vec);

#pragma unroll
      for (int i = 0; i < VecSize; ++i) {
        functor(x_vec[i],
                HasDX ? y_vec[i] : x_vec[i],
                dz_vec[i],
                &x_vec[i],
                &dz_vec[i]);
      }
      if constexpr (HasDX) {
        phi::Store<T, VecSize>(x_vec, dx + idx);
      }
      if constexpr (HasDY) {
        phi::Store<T, VecSize>(dz_vec, dy + idx);
      }
      idx += stride;
    }

    while (idx < numel) {
      functor(x[idx],
              HasDX ? y[idx] : x[idx],
              dz[idx],
              HasDX ? &dx[idx] : nullptr,
              HasDY ? &dy[idx] : nullptr);
      ++idx;
    }
  }
}

template <typename T, typename Context>
void SwiGLUGradKernelImpl(const Context &ctx,
                          const T *x,
                          const T *y,
                          const T *dz,
                          T *dx,
                          T *dy,
                          int64_t m,
                          int64_t n) {
  int vec_size =
      std::min(phi::GetVectorizedSize<T>(x), phi::GetVectorizedSize<T>(dz));
  if (y) {
    vec_size = std::min(vec_size, phi::GetVectorizedSize<T>(y));
  }
  if (dx) {
    vec_size = std::min(vec_size, phi::GetVectorizedSize<T>(dx));
  }
  if (dy) {
    vec_size = std::min(vec_size, phi::GetVectorizedSize<T>(dy));
  }

#define PD_LAUNCH_SWIGLU_GRAD_CUDA_KERNEL_BASE(                                \
    __vec_size, __is_combine, __has_dx, __has_dy)                              \
  case __vec_size: {                                                           \
    SwiGLUGradCUDAKernel<T, __vec_size, __is_combine, __has_dx, __has_dy>      \
        <<<config.block_per_grid, config.thread_per_block, 0, ctx.stream()>>>( \
            x, y, dz, dx, dy, m, n);                                           \
    break;                                                                     \
  }

#define PD_LAUNCH_SWIGLU_GRAD_CUDA_KERNEL(__is_combine, __has_dx, __has_dy) \
  do {                                                                      \
    switch (vec_size) {                                                     \
      PD_LAUNCH_SWIGLU_GRAD_CUDA_KERNEL_BASE(                               \
          VecSizeL, __is_combine, __has_dx, __has_dy);                      \
      PD_LAUNCH_SWIGLU_GRAD_CUDA_KERNEL_BASE(                               \
          VecSizeM, __is_combine, __has_dx, __has_dy);                      \
      PD_LAUNCH_SWIGLU_GRAD_CUDA_KERNEL_BASE(                               \
          VecSizeS, __is_combine, __has_dx, __has_dy);                      \
      default:                                                              \
        PADDLE_THROW(common::errors::Unimplemented(                         \
            "Unsupported vectorized size: %d !", vec_size));                \
        break;                                                              \
    }                                                                       \
  } while (0)

  if (y) {
    auto config =
        phi::backends::gpu::GetGpuLaunchConfig1D(ctx, m * n, vec_size);
    if (dx) {
      if (dy) {
        PD_LAUNCH_SWIGLU_GRAD_CUDA_KERNEL(false, true, true);
      } else {
        PD_LAUNCH_SWIGLU_GRAD_CUDA_KERNEL(false, true, false);
      }
    } else {
      PADDLE_ENFORCE_NOT_NULL(
          dy,
          common::errors::InvalidArgument(
              "Both gradients of Input(X) and Input(Y) is None."));
      PD_LAUNCH_SWIGLU_GRAD_CUDA_KERNEL(false, false, true);
    }
  } else {
    PADDLE_ENFORCE_NOT_NULL(
        dx,
        common::errors::InvalidArgument(
            "Both gradients of Input(X) and Input(Y) is None."));
    while (n % vec_size != 0) {
      vec_size /= 2;
    }
    y = x + n;
    dy = dx + n;
    auto config =
        phi::backends::gpu::GetGpuLaunchConfig1D(ctx, m * n / vec_size, 1);
    PD_LAUNCH_SWIGLU_GRAD_CUDA_KERNEL(true, true, true);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(swiglu_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SwiGLUGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
