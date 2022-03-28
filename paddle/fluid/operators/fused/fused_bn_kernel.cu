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

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/fluid/operators/fused/fused_bn_kernel.h"
#include "paddle/fluid/platform/device/gpu/cuda/cudnn_helper.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace paddle {
namespace operators {

template <typename T>
struct alignas(sizeof(T) * 2) SumPair {
  T x, y;

  HOSTDEVICE SumPair() {}

  HOSTDEVICE SumPair(T x, T y) : x(x), y(y) {}

  SumPair<T> &operator+=(const SumPair<T> &other) {
    x += other.x;
    y += other.y;
    return *this;
  }

  SumPair<T> operator+(const SumPair<T> &other) const {
    return SumPair<T>(x + other.x, y + other.y);
  }
};

template <typename T, int Size, bool IsAligned = true>
struct VectorizedArray : public phi::AlignedVector<T, Size> {
  static constexpr int kSize = Size;

  HOSTDEVICE void Load(const T *x) {
    phi::Load(x, static_cast<phi::AlignedVector<T, Size> *>(this));
  }

  HOSTDEVICE void Store(T *x) const {
    phi::Store(*static_cast<const phi::AlignedVector<T, Size> *>(this), x);
  }
};

template <typename T, int Size>
struct VectorizedArray<T, Size, false> {
  static constexpr int kSize = Size;

  T val[Size];

  HOSTDEVICE inline const T &operator[](int i) const { return val[i]; }
  HOSTDEVICE inline T &operator[](int i) { return val[i]; }

  HOSTDEVICE void Load(const T *x) {
#pragma unroll
    for (int i = 0; i < kSize; ++i) {
      val[i] = x[i];
    }
  }

  HOSTDEVICE void Store(T *x) const {
#pragma unroll
    for (int i = 0; i < kSize; ++i) {
      x[i] = val[i];
    }
  }
};

template <bool IsAligned, bool HasResidualAdd>
static __global__ void Vectorized8FP32MaskedReluFwdCUDAKernel(
    const float *x, const float *z, float *y, void *mask, size_t n) {
  size_t idx = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x) * 8;
  size_t stride = static_cast<size_t>(blockDim.x * gridDim.x) * 8;
  for (; idx + 8 <= n; idx += stride) {
    VectorizedArray<float, 8, IsAligned> x_vec, z_vec;
    auto *x_vec_ptr =
        reinterpret_cast<VectorizedArray<float, 4, IsAligned> *>(&x_vec);
    x_vec_ptr[0].Load(x + idx);
    x_vec_ptr[1].Load(x + idx + 4);

    if (HasResidualAdd) {
      auto *z_vec_ptr =
          reinterpret_cast<VectorizedArray<float, 4, IsAligned> *>(&z_vec);
      z_vec_ptr[0].Load(z + idx);
      z_vec_ptr[1].Load(z + idx + 4);
    }

    uint8_t mask_val = 0;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      auto tmp = HasResidualAdd ? (x_vec[i] + z_vec[i]) : x_vec[i];
      bool flag = (tmp > 0);
      x_vec[i] *= flag;
      mask_val |= (static_cast<uint8_t>(flag) << i);
    }
    x_vec_ptr[0].Store(y + idx);
    x_vec_ptr[1].Store(y + idx + 4);
    reinterpret_cast<uint8_t *>(mask)[idx / 8] = mask_val;
  }

  if (idx < n) {
    size_t left = n - idx;
    uint8_t mask_val = 0;
    for (size_t i = 0; i < left; ++i) {
      uint8_t mask_val = 0;
      auto tmp = HasResidualAdd ? (x[idx + i] + z[idx + i]) : x[idx + i];
      bool flag = (tmp > 0);
      y[idx + i] = tmp * flag;
      mask_val |= (static_cast<uint8_t>(flag) << i);
    }
    reinterpret_cast<uint8_t *>(mask)[idx / 8] = mask_val;
  }
}

template <bool IsAligned>
static __global__ void Vectorized8FP32MaskedReluBwdCUDAKernel(const float *dy,
                                                              const void *mask,
                                                              float *dx,
                                                              size_t n) {
  size_t idx = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x) * 8;
  size_t stride = static_cast<size_t>(blockDim.x * gridDim.x) * 8;
  for (; idx + 8 <= n; idx += stride) {
    uint8_t mask_val = reinterpret_cast<const uint8_t *>(mask)[idx / 8];
    VectorizedArray<float, 8, IsAligned> dy_vec;
    auto *dy_vec_ptr =
        reinterpret_cast<VectorizedArray<float, 4, IsAligned> *>(&dy_vec);
    dy_vec_ptr[0].Load(dy + idx);
    dy_vec_ptr[1].Load(dy + idx + 4);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      dy_vec[i] *= ((mask_val & (static_cast<uint8_t>(1) << i)) != 0);
    }
    dy_vec_ptr[0].Store(dx + idx);
    dy_vec_ptr[1].Store(dx + idx + 4);
  }

  if (idx < n) {
    size_t left = n - idx;
    uint8_t mask_val = reinterpret_cast<const uint8_t *>(mask)[idx / 8];
    for (size_t i = 0; i < left; ++i) {
      auto tmp = dy[idx + i];
      bool flag = ((mask_val & (static_cast<uint8_t>(1) << i)) != 0);
      dx[idx + i] = tmp * flag;
    }
  }
}

void LaunchFP32MaskedAddReluFwdKernel(
    const platform::CUDADeviceContext &dev_ctx, const float *x, const float *z,
    float *y, void *reserve_space, size_t n) {
  int vec_size = std::min(phi::GetVectorizedSize(x), phi::GetVectorizedSize(z));
  vec_size = std::min(vec_size, phi::GetVectorizedSize(z));
  auto config = platform::GetGpuLaunchConfig1D(dev_ctx, n, 8);
  auto stream = dev_ctx.stream();
#define LAUNCH_FP32_MASKED_ADD_RELU_FWD_KERNEL(__is_aligned, __has_residual) \
  do {                                                                       \
    Vectorized8FP32MaskedReluFwdCUDAKernel<__is_aligned, __has_residual><<<  \
        config.block_per_grid, config.thread_per_block, 0, stream>>>(        \
        x, z, y, reserve_space, n);                                          \
  } while (0)
  if (vec_size % 4 == 0) {
    if (z != nullptr) {
      LAUNCH_FP32_MASKED_ADD_RELU_FWD_KERNEL(true, true);
    } else {
      LAUNCH_FP32_MASKED_ADD_RELU_FWD_KERNEL(true, false);
    }
  } else {  // almost impossible case
    if (z != nullptr) {
      LAUNCH_FP32_MASKED_ADD_RELU_FWD_KERNEL(false, true);
    } else {
      LAUNCH_FP32_MASKED_ADD_RELU_FWD_KERNEL(false, false);
    }
  }
#undef LAUNCH_FP32_MASKED_ADD_RELU_FWD_KERNEL
}

void LaunchFP32MaskedReluBwdKernel(const platform::CUDADeviceContext &dev_ctx,
                                   const float *dy, const void *reserve_space,
                                   float *dx, size_t n) {
  int vec_size =
      std::min(phi::GetVectorizedSize(dx), phi::GetVectorizedSize(dy));
  auto config = platform::GetGpuLaunchConfig1D(dev_ctx, n, 8);
  auto stream = dev_ctx.stream();
  if (vec_size % 4 == 0) {
    Vectorized8FP32MaskedReluBwdCUDAKernel<
        true><<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
        dy, reserve_space, dx, n);
  } else {
    Vectorized8FP32MaskedReluBwdCUDAKernel<
        false><<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
        dy, reserve_space, dx, n);
  }
}

static size_t RoundUpFactor(size_t n, size_t m) { return (n + m - 1) / m; }

size_t GetFP32BNReserveSpaceSize(uint32_t N, uint32_t C, uint32_t H,
                                 uint32_t W) {
  size_t n = static_cast<size_t>(N) * C * H * W;
  return RoundUpFactor(n, 8);
}

bool CanUseFusedNCHWFP32BNTrainingKernel(uint32_t N, uint32_t C, uint32_t H,
                                         uint32_t W) {
  return static_cast<size_t>(H) * W % 8 == 0;
}

template <typename T>
struct BNStatusUpdater {
  BNStatusUpdater(const T *scale, const T *bias, T *save_mean,
                  T *save_inv_variance, T *running_mean, T *running_variance,
                  T *tmp_scale_bias, double factor, double epsilon)
      : scale(scale),
        bias(bias),
        save_mean(save_mean),
        save_inv_variance(save_inv_variance),
        running_mean(running_mean),
        running_variance(running_variance),
        tmp_scale_bias(tmp_scale_bias),
        factor(factor),
        epsilon(epsilon) {}

  template <bool IsAligned>
  HOSTDEVICE void Update(int idx, T mean_val, T variance_val) {
    save_mean[idx] = mean_val;
    T tmp_inv_var = 1.0f / sqrt(variance_val + epsilon);
    save_inv_variance[idx] = tmp_inv_var;
    T one_minus_factor = static_cast<T>(1) - factor;
    running_mean[idx] =
        one_minus_factor * running_mean[idx] + factor * mean_val;
    running_variance[idx] =
        one_minus_factor * running_variance[idx] + factor * variance_val;

    VectorizedArray<T, 2, IsAligned> tmp_scale_bias_vec;
    auto tmp_scale = scale[idx] * tmp_inv_var;
    tmp_scale_bias_vec[0] = tmp_scale;
    tmp_scale_bias_vec[1] = bias[idx] - mean_val * tmp_scale;
    tmp_scale_bias_vec.Store(tmp_scale_bias + 2 * idx);
  }

 private:
  const T *__restrict__ scale;
  const T *__restrict__ bias;
  T *__restrict__ save_mean;
  T *__restrict__ save_inv_variance;
  T *__restrict__ running_mean;
  T *__restrict__ running_variance;
  T *__restrict__ tmp_scale_bias;
  double factor;
  double epsilon;
};

template <typename T, int VecSize, int BlockDim, bool UpdateBNStatus,
          bool IsAligned>
static __global__ void BNMeanAndVarReduceStage1Kernel(
    const T *__restrict__ x, T *__restrict__ tmp_sum,
    T *__restrict__ tmp_square_sum, BNStatusUpdater<T> updater, size_t N,
    size_t C, size_t HW, size_t K, size_t mean_var_stride) {
  size_t c_idx = blockIdx.x / K;
  size_t k_idx = blockIdx.x % K;

  size_t NHW = N * HW;
  size_t CHW = C * HW;
  size_t stride = BlockDim * VecSize * K;
  size_t idx = (k_idx + threadIdx.x * K) * VecSize;
  SumPair<T> pair;
  pair.x = static_cast<T>(0);
  pair.y = static_cast<T>(0);
  for (; idx < NHW; idx += stride) {
    VectorizedArray<T, VecSize> x_vec;
    size_t x_idx = (idx / HW) * CHW + c_idx * HW + idx % HW;
    x_vec.Load(x + x_idx);
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      T tmp = x_vec[i];
      pair.x += tmp;
      pair.y += (tmp * tmp);
    }
  }

  using BlockReduce = cub::BlockReduce<SumPair<T>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage storage;
  pair = BlockReduce(storage).Reduce(pair, cub::Sum());
  if (threadIdx.x == 0) {
    size_t offset = c_idx * mean_var_stride + k_idx;
    if (!UpdateBNStatus) {
      tmp_sum[offset] = pair.x;
      tmp_square_sum[offset] = pair.y;
    } else {
      auto final_mean = pair.x / NHW;
      auto final_var = pair.y / NHW - final_mean * final_mean;
      updater.Update<IsAligned>(offset, final_mean, final_var);
    }
  }
}

template <typename T, int VecSize, int BlockDim, bool IsAligned>
static __global__ void BNMeanAndVarReduceStage2Kernel(
    const T *__restrict__ tmp_sum, const T *__restrict__ tmp_square_sum,
    BNStatusUpdater<T> updater, size_t C, size_t K, size_t NHW,
    uint32_t data_stride) {
  size_t idx = threadIdx.x * VecSize;
  size_t stride = BlockDim * VecSize;

  SumPair<T> pair;
  pair.x = static_cast<T>(0);
  pair.y = static_cast<T>(0);

  for (; idx + VecSize <= K; idx += stride) {
    auto real_idx = blockIdx.x * data_stride + idx;
    VectorizedArray<T, VecSize, IsAligned> tmp_sum_vec, tmp_square_sum_vec;
    tmp_sum_vec.Load(tmp_sum + real_idx);
    tmp_square_sum_vec.Load(tmp_square_sum + real_idx);
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      pair.x += tmp_sum_vec[i];
      pair.y += tmp_square_sum_vec[i];
    }
  }

  if (idx < K) {
    auto real_idx = blockIdx.x * data_stride + idx;
    VectorizedArray<T, VecSize, IsAligned> tmp_sum_vec, tmp_square_sum_vec;
    tmp_sum_vec.Load(tmp_sum + real_idx);
    tmp_square_sum_vec.Load(tmp_square_sum + real_idx);
    size_t diff = K - idx;
    for (size_t i = 0; i < diff; ++i) {
      pair.x += tmp_sum_vec[i];
      pair.y += tmp_square_sum_vec[i];
    }
  }

  using BlockReduce = cub::BlockReduce<SumPair<T>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage storage;
  pair = BlockReduce(storage).Reduce(pair, cub::Sum());
  if (threadIdx.x == 0) {
    auto final_mean = pair.x / NHW;
    auto final_var = pair.y / NHW - final_mean * final_mean;
    updater.Update<IsAligned>(blockIdx.x, final_mean, final_var);
  }
}

template <bool NeedResidual, bool NeedRelu, bool IsAligned>
static __global__ void BNFinalizeOutputKernel(const float *x, const float *z,
                                              const float *scale_bias, float *y,
                                              void *mask, size_t C, size_t HW,
                                              size_t NCHW) {
  size_t idx = (threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x) * 8;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x * 8;
  for (; idx < NCHW; idx += stride) {
    VectorizedArray<float, 8, IsAligned> x_vec, z_vec;
    auto *x_vec_ptr =
        reinterpret_cast<VectorizedArray<float, 4, IsAligned> *>(&x_vec);
    x_vec_ptr[0].Load(x + idx);
    x_vec_ptr[1].Load(x + idx + 4);
    if (NeedResidual) {
      auto *z_vec_ptr =
          reinterpret_cast<VectorizedArray<float, 4, IsAligned> *>(&z_vec);
      z_vec_ptr[0].Load(z + idx);
      z_vec_ptr[1].Load(z + idx + 4);
    }
    uint8_t mask_val = 0;
    size_t c_idx = (idx - idx % HW) / HW % C;
    VectorizedArray<float, 2, IsAligned> scale_bias_vec;
    scale_bias_vec.Load(scale_bias + c_idx * 2);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      auto tmp = NeedResidual ? (x_vec[i] + z_vec[i]) : x_vec[i];
      tmp = scale_bias_vec[0] * tmp + scale_bias_vec[1];

      if (NeedRelu) {
        bool flag = (tmp > 0);
        mask_val |= (static_cast<uint8_t>(flag) << i);
        x_vec[i] = tmp * flag;
      } else {
        x_vec[i] = tmp;
      }
    }
    x_vec_ptr[0].Load(y + idx);
    x_vec_ptr[1].Load(y + idx + 4);
    if (NeedRelu) {
      reinterpret_cast<uint8_t *>(mask)[idx / 8] = mask_val;
    }
  }
}

void LaunchFusedNCHWFP32BNTrainingKernel(
    const platform::CUDADeviceContext &dev_ctx, const float *x, const float *z,
    const float *scale, const float *bias, float *y, float *save_mean,
    float *save_inv_variance, float *running_mean, float *running_variance,
    void *reserve_space, uint32_t N, uint32_t C, uint32_t H, uint32_t W,
    double factor, double epsilon, bool need_relu) {
  constexpr int kBlockDim = 512;
  constexpr int kMaxReduceNum = 128;
  constexpr int kVecSize = 4;

  PADDLE_ENFORCE_EQ(CanUseFusedNCHWFP32BNTrainingKernel(N, C, H, W), false,
                    phi::errors::InvalidArgument(
                        "H(%d) * W(%d) should be exactly divided by 8.", H, W));
  framework::Tensor tmp_scale_bias_tensor;
  tmp_scale_bias_tensor.Resize({static_cast<int64_t>(C) * 2});
  auto *tmp_scale_bias = dev_ctx.Alloc<float>(&tmp_scale_bias_tensor);

  int vec_size = std::min(phi::GetVectorizedSize(x), phi::GetVectorizedSize(y));
  if (z != nullptr) {
    vec_size = std::min(vec_size, phi::GetVectorizedSize(z));
  }

  bool is_aligned = (vec_size % kVecSize == 0);
  is_aligned &= (reinterpret_cast<uintptr_t>(tmp_scale_bias) %
                     alignof(VectorizedArray<float, 2, true>) ==
                 0);

  size_t HW = static_cast<size_t>(H) * W;
  size_t NHW = N * HW;
  size_t K = RoundUpFactor(NHW, kMaxReduceNum / kVecSize * kBlockDim);
  K = std::min<size_t>(dev_ctx.GetCUDAMaxGridDimSize()[0] / C, K);

  size_t mean_var_stride = 1;
  float *tmp_mean = nullptr, *tmp_var = nullptr;
  framework::Tensor tmp_mean_tensor, tmp_var_tensor;
  if (K > 1) {
    mean_var_stride = RoundUpFactor(K, kVecSize) * kVecSize;
    tmp_mean = dev_ctx.Alloc<float>(&tmp_mean_tensor);
    tmp_var = dev_ctx.Alloc<float>(&tmp_var_tensor);
    is_aligned &= (phi::GetVectorizedSize(tmp_mean) % kVecSize == 0);
    is_aligned &= (phi::GetVectorizedSize(tmp_var) % kVecSize == 0);
  }

  auto stream = dev_ctx.stream();
  BNStatusUpdater<float> updater(scale, bias, save_mean, save_inv_variance,
                                 running_mean, running_variance, tmp_scale_bias,
                                 factor, epsilon);
#define LAUNCH_BN_MEAN_AND_VAR_STAGE2_KERNEL(__is_aligned, __block_dim) \
  case __block_dim: {                                                   \
    BNMeanAndVarReduceStage2Kernel<                                     \
        float, kVecSize, __block_dim,                                   \
        __is_aligned><<<C, __block_dim, 0, stream>>>(                   \
        tmp_mean, tmp_var, updater, C, K, NHW, mean_var_stride);        \
    break;                                                              \
  }

#define SWTICH_BN_MEAN_VAR_STAGE2(__is_aligned, __thread_num)               \
  switch (__thread_num) {                                                   \
    LAUNCH_BN_MEAN_AND_VAR_STAGE2_KERNEL(__is_aligned, 512);                \
    LAUNCH_BN_MEAN_AND_VAR_STAGE2_KERNEL(__is_aligned, 256);                \
    LAUNCH_BN_MEAN_AND_VAR_STAGE2_KERNEL(__is_aligned, 128);                \
    LAUNCH_BN_MEAN_AND_VAR_STAGE2_KERNEL(__is_aligned, 64);                 \
    LAUNCH_BN_MEAN_AND_VAR_STAGE2_KERNEL(__is_aligned, 32);                 \
    LAUNCH_BN_MEAN_AND_VAR_STAGE2_KERNEL(__is_aligned, 16);                 \
    LAUNCH_BN_MEAN_AND_VAR_STAGE2_KERNEL(__is_aligned, 8);                  \
    LAUNCH_BN_MEAN_AND_VAR_STAGE2_KERNEL(__is_aligned, 4);                  \
    LAUNCH_BN_MEAN_AND_VAR_STAGE2_KERNEL(__is_aligned, 2);                  \
    LAUNCH_BN_MEAN_AND_VAR_STAGE2_KERNEL(__is_aligned, 1);                  \
    default: {                                                              \
      PADDLE_THROW(phi::errors::InvalidArgument("Unexpected thread num %d", \
                                                __thread_num));             \
    }                                                                       \
  }

  if (K > 1) {
    size_t stage2_thread_num =
        std::min(kBlockDim, platform::RoundToPowerOfTwo(K));
    if (is_aligned) {
      BNMeanAndVarReduceStage1Kernel<float, kVecSize, kBlockDim, false,
                                     true><<<C * K, kBlockDim, 0, stream>>>(
          x, tmp_mean, tmp_var, updater, N, C, HW, K, mean_var_stride);
      SWTICH_BN_MEAN_VAR_STAGE2(true, stage2_thread_num);
    } else {
      BNMeanAndVarReduceStage1Kernel<float, kVecSize, kBlockDim, false,
                                     false><<<C * K, kBlockDim, 0, stream>>>(
          x, tmp_mean, tmp_var, updater, N, C, HW, K, mean_var_stride);
      SWTICH_BN_MEAN_VAR_STAGE2(false, stage2_thread_num);
    }
  } else {
    if (is_aligned) {
      BNMeanAndVarReduceStage1Kernel<float, kVecSize, kBlockDim, true,
                                     true><<<C * K, kBlockDim, 0, stream>>>(
          x, tmp_mean, tmp_var, updater, N, C, HW, K, mean_var_stride);
    } else {
      BNMeanAndVarReduceStage1Kernel<float, kVecSize, kBlockDim, true,
                                     false><<<C * K, kBlockDim, 0, stream>>>(
          x, tmp_mean, tmp_var, updater, N, C, HW, K, mean_var_stride);
    }
  }

  size_t NCHW = NHW * C;
  auto config = platform::GetGpuLaunchConfig1D(dev_ctx, NCHW, 8);
#define LAUNCH_BN_FINALIZE_OUTPUT_KERNEL(__need_residual, __need_relu,    \
                                         __is_aligned)                    \
  do {                                                                    \
    BNFinalizeOutputKernel<__need_residual, __need_relu, __is_aligned><<< \
        config.block_per_grid, config.thread_per_block, 0, stream>>>(     \
        x, z, tmp_scale_bias, y, reserve_space, C, HW, NCHW);             \
  } while (0)

  if (z != nullptr) {
    if (need_relu) {
      if (is_aligned) {
        LAUNCH_BN_FINALIZE_OUTPUT_KERNEL(true, true, true);
      } else {
        LAUNCH_BN_FINALIZE_OUTPUT_KERNEL(true, true, false);
      }
    } else {
      if (is_aligned) {
        LAUNCH_BN_FINALIZE_OUTPUT_KERNEL(true, false, true);
      } else {
        LAUNCH_BN_FINALIZE_OUTPUT_KERNEL(true, false, false);
      }
    }
  } else {
    if (need_relu) {
      if (is_aligned) {
        LAUNCH_BN_FINALIZE_OUTPUT_KERNEL(false, true, true);
      } else {
        LAUNCH_BN_FINALIZE_OUTPUT_KERNEL(false, true, false);
      }
    } else {
      if (is_aligned) {
        LAUNCH_BN_FINALIZE_OUTPUT_KERNEL(false, false, true);
      } else {
        LAUNCH_BN_FINALIZE_OUTPUT_KERNEL(false, false, false);
      }
    }
  }
}

}  // namespace operators
}  // namespace paddle
