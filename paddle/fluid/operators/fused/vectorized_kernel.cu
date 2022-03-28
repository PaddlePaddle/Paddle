// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/fused/vectorized_kernel.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace paddle {
namespace operators {

template <typename T, int N, int K>
static HOSTDEVICE void LargeLoad(const T *x, phi::AlignedVector<T, N> *x_vec) {
  static_assert(N % K == 0, "N % K must be 0");
  auto *x_vec_k = reinterpret_cast<phi::AlignedVector<T, K> *>(x_vec);
  for (int i = 0; i < N / K; ++i) {
    phi::Load(x + i * K, x_vec_k + i * K);
  }
}

template <typename T, int N, int K>
static HOSTDEVICE void LargeStore(const phi::AlignedVector<T, N> &x_vec, T *x) {
  static_assert(N % K == 0, "N % K must be 0");
  auto *x_vec_k = reinterpret_cast<const phi::AlignedVector<T, K> *>(&x_vec);
  for (int i = 0; i < N / K; ++i) {
    phi::Store(x_vec_k[i * K], x + i * K);
  }
}

static __global__ void Vectorized128MaskedReluFwdCUDAKernel(const float *x,
                                                            float *y,
                                                            void *mask,
                                                            size_t n) {
  size_t idx = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x) * 128;
  size_t stride = static_cast<size_t>(blockDim.x * gridDim.x) * 128;
  for (; idx < n; idx += stride) {
    phi::AlignedVector<uint64_t, 2> mask_vec;
    mask_vec[0] = 0;
    mask_vec[1] = 0;

    phi::AlignedVector<float, 64> x_vec;
    phi::Load(x + idx, &x_vec);
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      bool flag = (x_vec[i] > 0);
      x_vec[i] = fmaxf(x_vec[i], 0);
      mask_vec[0] |= (static_cast<uint64_t>(flag) << i);
    }
    phi::Store(x_vec, y + idx);

    phi::Load(x + idx + 64, &x_vec);
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      bool flag = (x_vec[i] > 0);
      x_vec[i] = fmaxf(x_vec[i], 0);
      mask_vec[1] |= (static_cast<uint64_t>(flag) << i);
    }
    phi::Store(x_vec, y + idx + 64);

    phi::Store(mask_vec, reinterpret_cast<uint64_t *>(mask) + idx / 64);
  }
}

static __global__ void Vectorized128MaskedAddReluFwdCUDAKernel(
    const float *x, const float *y, float *z, void *mask, size_t n) {
  size_t idx =
      (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 128;
  size_t stride = static_cast<size_t>(blockDim.x * gridDim.x) * 128;
  for (; idx < n; idx += stride) {
    phi::AlignedVector<uint64_t, 2> mask_vec;
    mask_vec[0] = 0;
    mask_vec[1] = 0;

    phi::AlignedVector<float, 32> x_vec, y_vec;

    phi::Load(x + idx, &x_vec);
    phi::Load(y + idx, &y_vec);
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      x_vec[i] += y_vec[i];

      bool flag = (x_vec[i] > 0);
      x_vec[i] = fmaxf(x_vec[i], 0);
      mask_vec[0] |= (static_cast<uint64_t>(flag) << i);
    }
    phi::Store(x_vec, z + idx);

    phi::Load(x + idx + 32, &x_vec);
    phi::Load(y + idx + 32, &y_vec);
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      x_vec[i] += y_vec[i];

      bool flag = (x_vec[i] > 0);
      x_vec[i] = fmaxf(x_vec[i], 0);
      mask_vec[0] |= (static_cast<uint64_t>(flag) << i);
    }
    phi::Store(x_vec, z + idx + 32);

    phi::Load(x + idx + 64, &x_vec);
    phi::Load(y + idx + 64, &y_vec);
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      x_vec[i] += y_vec[i];

      bool flag = (x_vec[i] > 0);
      x_vec[i] = fmaxf(x_vec[i], 0);
      mask_vec[1] |= (static_cast<uint64_t>(flag) << i);
    }
    phi::Store(x_vec, z + idx + 64);

    phi::Load(x + idx + 96, &x_vec);
    phi::Load(y + idx + 96, &y_vec);
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      x_vec[i] += y_vec[i];

      bool flag = (x_vec[i] > 0);
      x_vec[i] = fmaxf(x_vec[i], 0);
      mask_vec[1] |= (static_cast<uint64_t>(flag) << i);
    }
    phi::Store(x_vec, z + idx + 96);

    phi::Store(mask_vec, reinterpret_cast<uint64_t *>(mask) + idx / 64);
  }
}

static __global__ void Vectorized128MaskedReluBwdCUDAKernel(const float *dy,
                                                            const void *mask,
                                                            float *dx,
                                                            size_t n) {
  size_t idx = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x) * 128;
  size_t stride = static_cast<size_t>(blockDim.x * gridDim.x) * 128;
  for (; idx < n; idx += stride) {
    phi::AlignedVector<uint64_t, 2> mask_vec;
    phi::Load(reinterpret_cast<const uint64_t *>(mask) + idx / 64, &mask_vec);

    phi::AlignedVector<float, 64> dy_vec;
    phi::Load(dy + idx, &dy_vec);
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      bool flag = ((mask_vec[0] & (static_cast<uint64_t>(1) << i)) != 0);
      dy_vec[i] *= flag;
    }
    phi::Store(dy_vec, dx + idx);

    phi::Load(dy + idx + 64, &dy_vec);
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      bool flag = ((mask_vec[1] & (static_cast<uint64_t>(1) << i)) != 0);
      dy_vec[i] *= flag;
    }
    phi::Store(dy_vec, dx + idx + 64);
  }
}

static __global__ void Vectorized8MaskedReluFwdCUDAKernel(const float *x,
                                                          float *y, void *mask,
                                                          size_t n) {
  size_t idx = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x) * 8;
  size_t stride = static_cast<size_t>(blockDim.x * gridDim.x) * 8;
  for (; idx < n; idx += stride) {
    phi::AlignedVector<float, 8> x_vec;
    phi::Load(x + idx, &x_vec);
    uint8_t mask_val = 0;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      bool flag = (x_vec[i] > 0);
      x_vec[i] *= flag;
      mask_val |= (static_cast<uint8_t>(flag) << i);
    }
    phi::Store(x_vec, y + idx);
    reinterpret_cast<uint8_t *>(mask)[idx / 8] = mask_val;
  }
}

static __global__ void Vectorized8MaskedAddReluFwdCUDAKernel(
    const float *x, const float *y, float *z, void *mask, size_t n) {
  size_t idx = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x) * 8;
  size_t stride = static_cast<size_t>(blockDim.x * gridDim.x) * 8;
  for (; idx < n; idx += stride) {
    phi::AlignedVector<float, 8> x_vec, y_vec;
    phi::Load(x + idx, &x_vec);
    phi::Load(y + idx, &y_vec);
    uint8_t mask_val = 0;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      x_vec[i] += y_vec[i];
      bool flag = (x_vec[i] > 0);
      x_vec[i] *= flag;
      mask_val |= (static_cast<uint8_t>(flag) << i);
    }
    phi::Store(x_vec, z + idx);
    reinterpret_cast<uint8_t *>(mask)[idx / 8] = mask_val;
  }
}

static __global__ void Vectorized8MaskedReluBwdCUDAKernel(const float *dy,
                                                          const void *mask,
                                                          float *dx, size_t n) {
  size_t idx = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x) * 8;
  size_t stride = static_cast<size_t>(blockDim.x * gridDim.x) * 8;
  for (; idx < n; idx += stride) {
    uint8_t mask_val = reinterpret_cast<const uint8_t *>(mask)[idx / 8];
    phi::AlignedVector<float, 8> dy_vec;
    phi::Load(dy + idx, &dy_vec);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      dy_vec[i] *= ((mask_val & (static_cast<uint8_t>(1) << i)) != 0);
    }
    phi::Store(dy_vec, dx + idx);
  }
}

void LaunchVectorized128MaskedReluFwdKernel(
    const platform::CUDADeviceContext &dev_ctx, const float *x, float *y,
    void *mask, size_t n) {
  PADDLE_ENFORCE_EQ(reinterpret_cast<uintptr_t>(x) % 128, 0);
  PADDLE_ENFORCE_EQ(reinterpret_cast<uintptr_t>(y) % 128, 0);
  PADDLE_ENFORCE_EQ(reinterpret_cast<uintptr_t>(mask) % 128, 0);
  PADDLE_ENFORCE_EQ(n % 128, 0);

  /*
  auto config = platform::GetGpuLaunchConfig1D(dev_ctx, n, 128);
  auto stream = dev_ctx.stream();
  Vectorized128MaskedReluFwdCUDAKernel<<<
      config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(x, y,
                                                                       mask, n);
      */
  auto config = platform::GetGpuLaunchConfig1D(dev_ctx, n, 8);
  auto stream = dev_ctx.stream();
  Vectorized8MaskedReluFwdCUDAKernel<<<config.block_per_grid.x,
                                       config.thread_per_block.x, 0, stream>>>(
      x, y, mask, n);
}

void LaunchVectorized128MaskedAddReluFwdKernel(
    const platform::CUDADeviceContext &dev_ctx, const float *x, const float *y,
    float *z, void *mask, size_t n) {
  PADDLE_ENFORCE_EQ(reinterpret_cast<uintptr_t>(x) % 128, 0);
  PADDLE_ENFORCE_EQ(reinterpret_cast<uintptr_t>(y) % 128, 0);
  PADDLE_ENFORCE_EQ(reinterpret_cast<uintptr_t>(z) % 128, 0);
  PADDLE_ENFORCE_EQ(reinterpret_cast<uintptr_t>(mask) % 128, 0);
  PADDLE_ENFORCE_EQ(n % 128, 0);

  /*
  auto config = platform::GetGpuLaunchConfig1D(dev_ctx, n, 128);
  auto stream = dev_ctx.stream();
  Vectorized128MaskedAddReluFwdCUDAKernel<<<
      config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(x, y, z,
                                                                       mask, n);
  */
  auto config = platform::GetGpuLaunchConfig1D(dev_ctx, n, 8);
  auto stream = dev_ctx.stream();
  Vectorized8MaskedAddReluFwdCUDAKernel<<<
      config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(x, y, z,
                                                                       mask, n);
}

void LaunchVectorized128MaskedReluBwdKernel(
    const platform::CUDADeviceContext &dev_ctx, const float *dy,
    const void *mask, float *dx, size_t n) {
  PADDLE_ENFORCE_EQ(reinterpret_cast<uintptr_t>(dx) % 128, 0);
  PADDLE_ENFORCE_EQ(reinterpret_cast<uintptr_t>(dy) % 128, 0);
  PADDLE_ENFORCE_EQ(reinterpret_cast<uintptr_t>(mask) % 128, 0);

  /*
  auto config = platform::GetGpuLaunchConfig1D(dev_ctx, n, 128);
  auto stream = dev_ctx.stream();
  Vectorized128MaskedReluBwdCUDAKernel<<<
      config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(dy, mask,
                                                                       dx, n);
  */
  auto config = platform::GetGpuLaunchConfig1D(dev_ctx, n, 8);
  auto stream = dev_ctx.stream();
  Vectorized8MaskedReluBwdCUDAKernel<<<config.block_per_grid.x,
                                       config.thread_per_block.x, 0, stream>>>(
      dy, mask, dx, n);
}

}  // namespace operators
}  // namespace paddle
