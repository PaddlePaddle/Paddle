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

#pragma once

#include "glog/logging.h"
#include "paddle/fluid/memory/buffer.h"
#include "paddle/fluid/operators/norm_utils.h"
#include "paddle/fluid/platform/device/gpu/cuda/cudnn_helper.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/dynload/cudnn.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace paddle {
namespace operators {

template <typename T>
struct Pair {
  T x, y;

  HOSTDEVICE Pair() {}

  HOSTDEVICE Pair(int x, int y) : x(x), y(y) {}

  HOSTDEVICE Pair<T> &operator+=(const Pair<T> &other) {
    x += other.x;
    y += other.y;
    return *this;
  }

  HOSTDEVICE Pair<T> operator+(const Pair<T> &other) const {
    return Pair<T>(x + other.x, y + other.y);
  }
};

template <typename T>
struct BNStatusUpdater {
  BNStatusUpdater(const T *scale, const T *bias, T *save_mean,
                  T *save_inv_variance, T *running_mean, T *running_variance,
                  T *tmp_scale_bias, T factor, T epsilon)
      : scale(scale),
        bias(bias),
        save_mean(save_mean),
        save_inv_variance(save_inv_variance),
        running_mean(running_mean),
        running_variance(running_variance),
        tmp_scale_bias(tmp_scale_bias),
        factor(factor),
        epsilon(epsilon) {}

  HOSTDEVICE void Update(int idx, T mean_val, T variance_val) {
    save_mean[idx] = mean_val;
    T tmp_inv_var = 1.0f / sqrtf(variance_val + epsilon);
    save_inv_variance[idx] = tmp_inv_var;
    T one_minus_factor = static_cast<T>(1) - factor;
    running_mean[idx] =
        one_minus_factor * running_mean[idx] + factor * mean_val;
    running_variance[idx] =
        one_minus_factor * running_variance[idx] + factor * variance_val;

    phi::AlignedVector<T, 2> tmp_scale_bias_vec;
    auto tmp_scale = scale[idx] * tmp_inv_var;
    tmp_scale_bias_vec[0] = tmp_scale;
    tmp_scale_bias_vec[1] = bias[idx] - mean_val * tmp_scale;
    phi::Store(tmp_scale_bias_vec, tmp_scale_bias + 2 * idx);
  }

 private:
  const T *__restrict__ scale;
  const T *__restrict__ bias;
  T *__restrict__ save_mean;
  T *__restrict__ save_inv_variance;
  T *__restrict__ running_mean;
  T *__restrict__ running_variance;
  T *__restrict__ tmp_scale_bias;
  T factor;
  T epsilon;
};

template <typename T, int VecSize, int BlockDim, bool UpdateBNStatus>
static __global__ void BNMeanAndSquareMeanKernel(
    const T *__restrict__ x, T *__restrict__ tmp_sum,
    T *__restrict__ tmp_square_sum, BNStatusUpdater<T> updater, uint32_t N,
    uint32_t C, uint32_t HW, uint32_t K, uint32_t mean_var_stride) {
  uint32_t c_idx = blockIdx.x / K;
  uint32_t k_idx = blockIdx.x % K;

  uint32_t NHW = N * HW;
  uint32_t CHW = C * HW;
  uint32_t stride = BlockDim * VecSize * K;
  uint32_t idx = (k_idx + threadIdx.x * K) * VecSize;
  Pair<T> pair;
  pair.x = static_cast<T>(0);
  pair.y = static_cast<T>(0);
  for (; idx < NHW; idx += stride) {
    phi::AlignedVector<T, VecSize> x_vec;
    int x_idx = (idx / HW) * CHW + c_idx * HW + idx % HW;
    phi::Load(x + x_idx, &x_vec);
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      T tmp = x_vec[i];
      pair.x += tmp;
      pair.y += (tmp * tmp);
    }
  }

  using BlockReduce = cub::BlockReduce<Pair<T>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage storage;
  pair = BlockReduce(storage).Reduce(pair, cub::Sum());
  if (threadIdx.x == 0) {
    int offset = c_idx * mean_var_stride + k_idx;
    if (!UpdateBNStatus) {
      tmp_sum[offset] = pair.x;
      tmp_square_sum[offset] = pair.y;
      // printf("sum = %f, square_sum = %f, K = %u\n", pair.x, pair.y, K);
    } else {
      auto final_mean = pair.x / NHW;
      auto final_var = pair.y / NHW - final_mean * final_mean;
      updater.Update(offset, final_mean, final_var);
    }
  }
}

template <typename T, int VecSize, int BlockDim>
static __global__ void BNMeanAndSquareMeanKernel2(
    const T *__restrict__ tmp_sum, const T *__restrict__ tmp_square_sum,
    BNStatusUpdater<T> updater, uint32_t C, uint32_t K, uint32_t NHW,
    uint32_t data_stride) {
  uint32_t idx = threadIdx.x * VecSize;
  uint32_t stride = BlockDim * VecSize;

  Pair<T> pair;
  pair.x = static_cast<T>(0);
  pair.y = static_cast<T>(0);

  for (; idx + VecSize <= K; idx += stride) {
    auto real_idx = blockIdx.x * data_stride + idx;
    phi::AlignedVector<T, VecSize> tmp_sum_vec, tmp_square_sum_vec;
    phi::Load(tmp_sum + real_idx, &tmp_sum_vec);
    phi::Load(tmp_square_sum + real_idx, &tmp_square_sum_vec);
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      pair.x += tmp_sum_vec[i];
      pair.y += tmp_square_sum_vec[i];
    }
  }

  if (idx < K) {
    auto real_idx = blockIdx.x * data_stride + idx;
    phi::AlignedVector<T, VecSize> tmp_sum_vec, tmp_square_sum_vec;
    phi::Load(tmp_sum + real_idx, &tmp_sum_vec);
    phi::Load(tmp_square_sum + real_idx, &tmp_square_sum_vec);
    int diff = K - idx;
    for (int i = 0; i < diff; ++i) {
      pair.x += tmp_sum_vec[i];
      pair.y += tmp_square_sum_vec[i];
    }
  }

  using BlockReduce = cub::BlockReduce<Pair<T>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage storage;
  pair = BlockReduce(storage).Reduce(pair, cub::Sum());
  if (threadIdx.x == 0) {
    auto final_mean = pair.x / NHW;
    auto final_var = pair.y / NHW - final_mean * final_mean;
    updater.Update(blockIdx.x, final_mean, final_var);
  }
}

template <typename T>
static bool LaunchBNFastStatusKernel(const platform::CUDADeviceContext &dev_ctx,
                                     const T *x, const T *scale, const T *bias,
                                     T *save_mean, T *save_inv_variance,
                                     T *running_mean, T *running_variance,
                                     T factor, T epsilon, uint32_t N,
                                     uint32_t C, uint32_t H, uint32_t W,
                                     memory::Buffer *buffer) {
  constexpr int kVecSize = 4;
  if (N * W % kVecSize != 0) return false;

  int max_reduce_num = 128;

  BNStatusUpdater<T> updater(scale, bias, save_mean, save_inv_variance,
                             running_mean, running_variance,
                             buffer->Alloc<T>(2 * C), factor, epsilon);
  auto NHW = N * H * W;

  int thread_num = 512;
  auto K = platform::DivUp(NHW, max_reduce_num / kVecSize * thread_num);

  memory::Buffer tmp_mean(dev_ctx.GetPlace()), tmp_var(dev_ctx.GetPlace());
  uint32_t mean_var_stride = 1;
  T *tmp_mean_ptr = nullptr, *tmp_var_ptr = nullptr;
  if (K > 1) {
    mean_var_stride = platform::DivUp(K, 4) * 4;
    tmp_mean_ptr = tmp_mean.Alloc<T>(C * mean_var_stride);
    tmp_var_ptr = tmp_var.Alloc<T>(C * mean_var_stride);
  }
  auto stream = dev_ctx.stream();

#define LAUNCH_BN_STATUS_KERNEL(__vec_size, __block_dim)                    \
  case __block_dim: {                                                       \
    do {                                                                    \
      if (K > 1) {                                                          \
        BNMeanAndSquareMeanKernel<T, __vec_size, __block_dim,               \
                                  false><<<C * K, thread_num, 0, stream>>>( \
            x, tmp_mean_ptr, tmp_var_ptr, updater, N, C, H * W, K,          \
            mean_var_stride);                                               \
      } else {                                                              \
        BNMeanAndSquareMeanKernel<T, __vec_size, __block_dim,               \
                                  true><<<C * K, thread_num, 0, stream>>>(  \
            x, tmp_mean_ptr, tmp_var_ptr, updater, N, C, H * W, K,          \
            mean_var_stride);                                               \
      }                                                                     \
    } while (0);                                                            \
    break;                                                                  \
  }

  switch (thread_num) {
    LAUNCH_BN_STATUS_KERNEL(kVecSize, 512);
    LAUNCH_BN_STATUS_KERNEL(kVecSize, 256);
    LAUNCH_BN_STATUS_KERNEL(kVecSize, 128);
    LAUNCH_BN_STATUS_KERNEL(kVecSize, 64);
    LAUNCH_BN_STATUS_KERNEL(kVecSize, 32);
    LAUNCH_BN_STATUS_KERNEL(kVecSize, 16);
    LAUNCH_BN_STATUS_KERNEL(kVecSize, 4);
    LAUNCH_BN_STATUS_KERNEL(kVecSize, 2);
    LAUNCH_BN_STATUS_KERNEL(kVecSize, 1);
    default:
      PADDLE_THROW("unexpected branch");
  }

  if (K > 1) {
    int thread_num = std::min<uint32_t>(512, platform::RoundToPowerOfTwo(K));
#define LAUNCH_BN_STATUS_KERNEL_2(__vec_size, __block_dim)                   \
  case __block_dim: {                                                        \
    do {                                                                     \
      BNMeanAndSquareMeanKernel2<T, __vec_size,                              \
                                 __block_dim><<<C, thread_num, 0, stream>>>( \
          tmp_mean_ptr, tmp_var_ptr, updater, C, K, N * H * W,               \
          mean_var_stride);                                                  \
    } while (0);                                                             \
    break;                                                                   \
  }

    switch (thread_num) {
      LAUNCH_BN_STATUS_KERNEL_2(kVecSize, 512);
      LAUNCH_BN_STATUS_KERNEL_2(kVecSize, 256);
      LAUNCH_BN_STATUS_KERNEL_2(kVecSize, 128);
      LAUNCH_BN_STATUS_KERNEL_2(kVecSize, 64);
      LAUNCH_BN_STATUS_KERNEL_2(kVecSize, 32);
      LAUNCH_BN_STATUS_KERNEL_2(kVecSize, 16);
      LAUNCH_BN_STATUS_KERNEL_2(kVecSize, 4);
      LAUNCH_BN_STATUS_KERNEL_2(kVecSize, 2);
      LAUNCH_BN_STATUS_KERNEL_2(kVecSize, 1);
      default:
        PADDLE_THROW("unexpected branch");
    }
  }
  return true;
}

template <typename T, bool NeedResidual, bool NeedRelu>
static __global__ void BNFinalizeOutputKernel(const T *x, const T *z,
                                              const T *scale_bias, T *y,
                                              void *mask, uint32_t C,
                                              uint32_t HW, uint32_t NCHW) {
  uint32_t idx = (threadIdx.x + blockIdx.x * blockDim.x) * 8;
  uint32_t stride = blockDim.x * gridDim.x * 8;
  for (; idx < NCHW; idx += stride) {
    phi::AlignedVector<T, 8> x_vec, z_vec;
    phi::Load(x + idx, &x_vec);
    if (NeedResidual) {
      phi::Load(z + idx, &z_vec);
    }
    uint8_t mask_val = 0;
    uint32_t c_idx = (idx - idx % HW) / HW % C;
    phi::AlignedVector<T, 2> scale_bias_vec;
    phi::Load(scale_bias + c_idx * 2, &scale_bias_vec);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      T tmp = NeedResidual ? (x_vec[i] + z_vec[i]) : x_vec[i];
      tmp = scale_bias_vec[0] * tmp + scale_bias_vec[1];

      if (NeedRelu) {
        bool flag = (tmp > 0);
        mask_val |= (static_cast<uint8_t>(flag) << i);
        x_vec[i] = fmaxf(tmp, 0);
      } else {
        x_vec[i] = tmp;
      }
    }
    phi::Store(x_vec, y + idx);
    if (NeedRelu) {
      reinterpret_cast<uint8_t *>(mask)[idx / 8] = mask_val;
    }
  }
}

template <typename T>
static bool BNFinalizeOutput(const platform::CUDADeviceContext &dev_ctx,
                             const T *x, const T *z, const T *scale_bias, T *y,
                             void *mask, bool need_relu, uint32_t N, uint32_t C,
                             uint32_t H, uint32_t W) {
  if (N * C * H * W % 8 != 0) return false;

#define LAUNCH_BNFinalizeOutputKernel(__need_residual, __need_relu)          \
  do {                                                                       \
    auto config = platform::GetGpuLaunchConfig1D(dev_ctx, N * C * H * W, 8); \
    BNFinalizeOutputKernel<                                                  \
        T, __need_residual,                                                  \
        __need_relu><<<config.block_per_grid, config.thread_per_block, 0,    \
                       dev_ctx.stream()>>>(x, z, scale_bias, y, mask, C,     \
                                           H * W, N * C * H * W);            \
  } while (0)

  if (z != nullptr) {
    if (need_relu) {
      LAUNCH_BNFinalizeOutputKernel(true, true);
    } else {
      LAUNCH_BNFinalizeOutputKernel(true, false);
    }
  } else {
    if (need_relu) {
      LAUNCH_BNFinalizeOutputKernel(false, true);
    } else {
      LAUNCH_BNFinalizeOutputKernel(false, false);
    }
  }
  return true;
}

template <typename T>
static bool LaunchFastBNKernel(const platform::CUDADeviceContext &dev_ctx,
                               const T *x, const T *z, const T *scale,
                               const T *bias, T *y, T *save_mean,
                               T *save_inv_variance, T *running_mean,
                               T *running_variance, void *reserve_space,
                               uint32_t N, uint32_t C, uint32_t H, uint32_t W,
                               T factor, T epsilon, bool need_relu) {
  if (H * W % 8 != 0) return false;

  memory::Buffer buffer(dev_ctx.GetPlace());
  PADDLE_ENFORCE_EQ(LaunchBNFastStatusKernel(dev_ctx, x, scale, bias, save_mean,
                                             save_inv_variance, running_mean,
                                             running_variance, factor, epsilon,
                                             N, C, H, W, &buffer),
                    true);

  auto *scale_bias = buffer.Get<T>();
  PADDLE_ENFORCE_EQ(BNFinalizeOutput(dev_ctx, x, z, scale_bias, y,
                                     reserve_space, need_relu, N, C, H, W),
                    true);
  return true;
}

template <typename T>
static bool LaunchFP32FastBNKernel(const platform::CUDADeviceContext &dev_ctx,
                                   const void *x, const void *z,
                                   const void *scale, const void *bias, void *y,
                                   void *save_mean, void *save_inv_variance,
                                   void *running_mean, void *running_variance,
                                   void *reserve_space, uint32_t N, uint32_t C,
                                   uint32_t H, uint32_t W, float factor,
                                   float epsilon, bool need_relu) {
  if (!std::is_same<T, float>::value) return false;
  return LaunchFastBNKernel<float>(
      dev_ctx, static_cast<const float *>(x), static_cast<const float *>(z),
      static_cast<const float *>(scale), static_cast<const float *>(bias),
      static_cast<float *>(y), static_cast<float *>(save_mean),
      static_cast<float *>(save_inv_variance),
      static_cast<float *>(running_mean),
      static_cast<float *>(running_variance), reserve_space, N, C, H, W,
      static_cast<T>(factor), static_cast<float>(epsilon), need_relu);
}

}  // namespace operators
}  // namespace paddle
