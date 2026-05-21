// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/legacy/gpu/fp8_quant_blockwise_kernel.h"
#include <cuda_fp8.h>
#include <cstdint>
#include <vector>
#include "paddle/common/flags.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/utils/data_type.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fusion/gpu/quant_utils.h"

#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace phi {
constexpr size_t k_block_span = 128;
constexpr size_t k_threads_per_warp = 32;
constexpr size_t k_warp_xdim_size = 64;
constexpr size_t k_warp_ydim_size = 32;
constexpr size_t k_thread_dim_size = 8;

template <typename T>
struct CudaTypeTraits {
  using Type = T;
};

template <>
struct CudaTypeTraits<phi::bfloat16> {
  using Type = __nv_bfloat16;
};

template <>
struct CudaTypeTraits<phi::float16> {
  using Type = __half;
};

template <typename T>
struct alignas(16) v128_t {
  union data_t {
    T scalar[8];
    uint4 vector;  // 128-bit vector for 8x bfloat16
  };
  data_t data;

  __device__ __forceinline__ void load(const void *ptr) {
    data.vector = *reinterpret_cast<const uint4 *>(ptr);
  }

  __device__ __forceinline__ void store(void *ptr) const {
    *reinterpret_cast<uint4 *>(ptr) = data.vector;
  }
};

template <typename T>
struct alignas(8) v64_t {
  T val[4];
};

typedef struct alignas(16) fp8x8_t {
  union data_t {
    __nv_fp8_e4m3 scalar[8];
    uint2 vector;
  };
  data_t data;

  __device__ __forceinline__ void load(const void *ptr) {
    data.vector = *reinterpret_cast<const uint2 *>(ptr);
  }

  __device__ __forceinline__ void store(void *ptr) const {
    *reinterpret_cast<uint2 *>(ptr) = data.vector;
  }
} fp8x8_t;

struct alignas(4) fp8x4_t {
  __nv_fp8_e4m3 val[4];
};

__device__ __forceinline__ __half device_abs(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __habs(x);
#else
  return __float2half(fabsf(__half2float(x)));
#endif
}

__device__ __forceinline__ __nv_bfloat16 device_abs(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __habs(x);
#else
  return __float2bfloat16(fabsf(__bfloat162float(x)));
#endif
}

__device__ __forceinline__ float device_abs(float x) { return fabsf(x); }

__device__ __forceinline__ __half device_max(__half a, __half b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __hmax(a, b);
#else
  return __float2half(fmaxf(__half2float(a), __half2float(b)));
#endif
}

__device__ __forceinline__ __nv_bfloat16 device_max(__nv_bfloat16 a,
                                                    __nv_bfloat16 b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __hmax(a, b);
#else
  return __float2bfloat16(fmaxf(__bfloat162float(a), __bfloat162float(b)));
#endif
}

__device__ __forceinline__ float device_max(float a, float b) {
  return fmaxf(a, b);
}

template <bool Power2Scaling>
__device__ __forceinline__ float ScaleWrapper(const float amax,
                                              const float eps) {
  constexpr float fp8_max = 448.0f;
  float amax_mod = fmaxf(amax, eps);
  if (amax_mod == 0.f) {
    return 1.f;
  }
  float scale = fp8_max / amax_mod;

  if (isinf(scale)) {
    if constexpr (Power2Scaling) {
      return 0x1.0p127;
    } else {
      return 0x1.FEp127;
    }
  }

  if (scale == 0.0) {
    return scale;
  }
  if constexpr (Power2Scaling) {
    uint32_t scale_bits = *reinterpret_cast<uint32_t *>(&scale);
    uint8_t exp = scale_bits >> 23;
    int32_t normal_biased_exp = static_cast<int32_t>(exp) - 127;
    __builtin_assume(exp != 0);
    scale = ldexpf(1.0f, normal_biased_exp);
  }
  return scale;
}

template <typename ScaleT, bool using_ue8m0_scale>
__device__ __forceinline__ void StoreScale(ScaleT *ptr, size_t idx, float val) {
  if constexpr (using_ue8m0_scale) {
    int exp = (__float_as_int(val) >> 23) & 0xFF;
    reinterpret_cast<uint8_t *>(ptr)[idx] = static_cast<uint8_t>(exp);
  } else {
    ptr[idx] = val;
  }
}

template <typename T,
          typename ScaleT,
          bool input_transpose,
          bool output_scale_transpose,
          bool return_transpose_only,
          bool use_pow2_scale,
          bool using_ue8m0_scale>
__global__ void __launch_bounds__(256)
    quantize_128x128_kernel(const T *const input,
                            __nv_fp8_e4m3 *const output,
                            __nv_fp8_e4m3 *const output_transposed,
                            ScaleT *const scale,
                            ScaleT *const scale_transposed,
                            const size_t cols,
                            const size_t rows,
                            const size_t quanted_cols,
                            const size_t quanted_rows,
                            const float epsilon) {
  const size_t col_blocks = cols / k_block_span;
  const size_t block_x = blockIdx.x % col_blocks;
  const size_t block_y = blockIdx.x / col_blocks;
  constexpr size_t k_warpnum_x = k_block_span / k_warp_xdim_size;
  constexpr size_t k_warpnum_y = k_block_span / k_warp_ydim_size;
  constexpr size_t k_warpnum_total = k_warpnum_x * k_warpnum_y;
  constexpr size_t k_warp_span_x = k_warp_xdim_size / k_thread_dim_size;
  constexpr size_t k_warp_span_y = k_threads_per_warp / k_warp_span_x;
  constexpr size_t scale_stride_x = 1;
  constexpr size_t scale_t_stride_x = 1;
  const size_t scale_stride_y =
      using_ue8m0_scale ? quanted_cols * 4 : quanted_cols;
  const size_t scale_t_stride_y =
      using_ue8m0_scale ? quanted_rows * 4 : quanted_rows;
  const unsigned int lane = threadIdx.x % k_threads_per_warp;
  const unsigned int tid_in_warp_x = lane % k_warp_span_x;
  const unsigned int tid_in_warp_y = lane / k_warp_span_x;
  const unsigned int warp_id_in_block = threadIdx.x / k_threads_per_warp;
  const unsigned int warp_id_in_block_x = warp_id_in_block % k_warpnum_x;
  const unsigned int warp_id_in_block_y = warp_id_in_block / k_warpnum_x;

  float blockwise_amax;   // Uninitialized
  float blockwise_scale;  // Uninitialized
  float warpwise_amax;    // Uninitialized
  float thread_local_amax = 0.0f;

  v128_t<T> thread_local_input[k_thread_dim_size];  // NOLINT
  fp8x8_t thread_local_output;
  fp8x8_t thread_local_output_transposed[input_transpose ? k_thread_dim_size
                                                         : 1];  // NOLINT

  __shared__ float shared_warp_amax[k_warpnum_total];

  const size_t block_base_idx =
      block_y * k_block_span * cols + block_x * k_block_span;
  const size_t warp_base_idx =
      block_base_idx +
      warp_id_in_block_y * k_thread_dim_size * k_warp_span_y * cols +
      warp_id_in_block_x * k_thread_dim_size * k_warp_span_x;
  const size_t thread_base_idx = warp_base_idx +
                                 tid_in_warp_y * k_thread_dim_size * cols +
                                 tid_in_warp_x * k_thread_dim_size;
  const size_t block_transpose_base_idx =
      input_transpose ? block_x * k_block_span * rows + block_y * k_block_span
                      : -1;
  const size_t warp_transposed_base_idx =
      input_transpose
          ? block_transpose_base_idx +
                warp_id_in_block_x * k_thread_dim_size * k_warp_span_x * rows +
                warp_id_in_block_y * k_thread_dim_size * k_warp_span_y
          : -1;
  const size_t thread_transposed_base_idx =
      input_transpose ? warp_transposed_base_idx +
                            tid_in_warp_x * k_thread_dim_size * rows +
                            tid_in_warp_y * k_thread_dim_size
                      : -1;

#pragma unroll
  for (int i = 0; i < k_thread_dim_size; i++) {
    thread_local_input[i].load(input + thread_base_idx + i * cols);
  }

  for (int i = 0; i < k_thread_dim_size; i++) {
#pragma unroll
    for (int j = 0; j < k_thread_dim_size; j++) {
      thread_local_amax = fmaxf(
          thread_local_amax,
          fabsf(static_cast<float>(thread_local_input[i].data.scalar[j])));
    }
  }
  float warpwise_amax_proposed = thread_local_amax;
#pragma unroll
  for (int offset = 32; offset > 0; offset /= 2) {
    float received_max =
        __shfl_down_sync(0xFFFFFFFF, warpwise_amax_proposed, offset);
    warpwise_amax_proposed = fmaxf(warpwise_amax_proposed, received_max);
  }

  if (lane == 0) {
    shared_warp_amax[warp_id_in_block_y * k_warpnum_x + warp_id_in_block_x] =
        warpwise_amax_proposed;
  }

  __syncthreads();
  static_assert(k_warpnum_total == 8, "warpnum_total must be 8");
  if (threadIdx.x < 8) {
    float blockwise_amax_proposed = shared_warp_amax[threadIdx.x];
#pragma unroll
    for (int offset = 4; offset > 0; offset /= 2) {
      float received_max =
          __shfl_down_sync(0xFF, blockwise_amax_proposed, offset);
      blockwise_amax_proposed = fmaxf(blockwise_amax_proposed, received_max);
    }
    if (threadIdx.x == 0) {
      shared_warp_amax[0] = blockwise_amax_proposed;
    }
  }
  __syncthreads();
  blockwise_amax = shared_warp_amax[0];

  // 1. Unified scale calculation: Always compute multiplication scale
  blockwise_scale = ScaleWrapper < using_ue8m0_scale ||
                    use_pow2_scale > (blockwise_amax, epsilon);

  // 2. Compute store scale (inverse of multiplication scale)
  float store_scale = 1.0f / blockwise_scale;
  if constexpr (using_ue8m0_scale) {
    if (threadIdx.x < 128) {
      size_t row_idx = block_y * 128 + static_cast<size_t>(threadIdx.x);
      size_t col_idx = block_x;
      if constexpr (!return_transpose_only) {
        size_t idx;
        idx = output_scale_transpose
                  ? (col_idx / 4) * (rows * 4) + row_idx * 4 + (col_idx % 4)
                  : row_idx * scale_stride_y + col_idx;
        StoreScale<ScaleT, using_ue8m0_scale>(scale, idx, store_scale);
      }

      if constexpr (input_transpose) {
        row_idx = block_x * 128 + threadIdx.x;
        col_idx = block_y;
        size_t idx;
        // 4. Whether it is 1*128 or 128*128, when output_scale_transpose is
        // True and using_ue8m0_scale is enabled, the calculation of idx needs
        // to be independently upgraded, because 4 original floats are packed as
        // a group and then transposed. If the transposed idx is calculated
        // directly, the data will be wrong.
        idx = output_scale_transpose
                  ? (col_idx / 4) * (cols * 4) + row_idx * 4 + (col_idx % 4)
                  : row_idx * scale_t_stride_y + col_idx;
        StoreScale<ScaleT, using_ue8m0_scale>(
            scale_transposed, idx, store_scale);
      }
    }
  } else {
    if (threadIdx.x == 0) {
      size_t row_idx = block_y;
      size_t col_idx = block_x;
      if constexpr (!return_transpose_only) {
        size_t idx = output_scale_transpose
                         ? col_idx * scale_stride_y + row_idx * scale_stride_x
                         : row_idx * scale_stride_y + col_idx * scale_stride_x;
        StoreScale<ScaleT, using_ue8m0_scale>(scale, idx, store_scale);
      }

      if constexpr (input_transpose) {
        row_idx = block_x;
        col_idx = block_y;
        size_t idx =
            output_scale_transpose
                ? col_idx * scale_t_stride_y + row_idx * scale_t_stride_x
                : row_idx * scale_t_stride_y + col_idx * scale_t_stride_x;
        StoreScale<ScaleT, using_ue8m0_scale>(
            scale_transposed, idx, store_scale);
      }
    }
  }

  for (int i = 0; i < k_thread_dim_size; i++) {
#pragma unroll
    for (int j = 0; j < k_thread_dim_size; j++) {
      __nv_fp8_e4m3 output_fp8 = static_cast<__nv_fp8_e4m3>(
          static_cast<float>(thread_local_input[i].data.scalar[j]) *
          blockwise_scale);
      if constexpr (!return_transpose_only) {
        thread_local_output.data.scalar[j] = output_fp8;
      }
      if constexpr (input_transpose) {
        thread_local_output_transposed[j].data.scalar[i] = output_fp8;
      }
    }
    if constexpr (!return_transpose_only) {
      thread_local_output.store(output + thread_base_idx + i * cols);
    }
  }
  if constexpr (input_transpose) {
#pragma unroll
    for (int i = 0; i < k_thread_dim_size; i++) {
      thread_local_output_transposed[i].store(
          output_transposed + thread_transposed_base_idx + i * rows);
    }
  }
}

template <typename T, bool use_pow2_scale, bool using_ue8m0_scale>
__device__ void ComputeColumnScale(const v64_t<T> x[8],
                                   float block_scale[128],
                                   T *shm,
                                   const float epsilon) {
  // reduce [(8), 16, 32, 4] => [16, 32, 4]
  T local_max[4];
  for (uint32_t i = 0; i < 8; i++) {
    for (uint32_t j = 0; j < 4; j++) {
      T val = device_abs(x[i].val[j]);
      local_max[j] = i == 0 ? val : device_max(val, local_max[j]);
    }
  }

  // reduce [(16), 32, 4] => [8, 32, 4]
  if (threadIdx.y >= 8) {
    for (uint32_t j = 0; j < 4; j++) {
      shm[(threadIdx.y - 8) * 128 + threadIdx.x + j * 32] = local_max[j];
    }
  }
  __syncthreads();

  // reduce [(8), 32, 4] => [32, 4]
  for (uint32_t offset = 8; offset > 0; offset /= 2) {
    if (threadIdx.y < offset) {
      for (uint32_t j = 0; j < 4; j++) {
        T other =
            offset == 8
                ? local_max[j]
                : shm[(threadIdx.y + offset) * 128 + threadIdx.x + j * 32];
        T new_val =
            device_max(shm[threadIdx.y * 128 + threadIdx.x + j * 32], other);
        if (offset > 1) {
          shm[threadIdx.y * 128 + threadIdx.x + j * 32] = new_val;
        } else {
          block_scale[threadIdx.x * 4 + j] =
              ScaleWrapper < using_ue8m0_scale ||
              use_pow2_scale > (static_cast<float>(new_val), epsilon);
        }
      }
    }
    __syncthreads();
  }
}

template <typename T, bool use_pow2_scale, bool using_ue8m0_scale>
__device__ void ComputeRowScale(const v64_t<T> x[8],
                                float block_scale[128],
                                T *shm,
                                const float epsilon) {
  for (uint32_t i = 0; i < 8; i++) {
    // reduce [32, (4)] => [32]
    T local_max;
    for (uint32_t j = 0; j < 4; j++) {
      T other = device_abs(x[i].val[j]);
      local_max = j == 0 ? other : device_max(local_max, other);
    }

    // reduce [32] => [1]
    T warp_max = local_max;
    for (uint32_t offset = 16; offset > 0; offset /= 2) {
      T other = __shfl_down_sync(0xFFFFFFFF, warp_max, offset);
      warp_max = device_max(warp_max, other);
    }
    if (threadIdx.x == 0) {
      shm[i * 16 + threadIdx.y] = warp_max;
    }
  }
  __syncthreads();

  // compute scale
  if (threadIdx.y < 4) {
    T amax = shm[threadIdx.y * 32 + threadIdx.x];
    block_scale[threadIdx.y * 32 + threadIdx.x] =
        ScaleWrapper < using_ue8m0_scale ||
        use_pow2_scale > (static_cast<float>(amax), epsilon);
  }
  __syncthreads();
}

template <typename T,
          typename ScaleT,
          bool input_transpose,
          bool output_scale_transpose,
          bool return_transpose_only,
          bool use_pow2_scale,
          bool using_ue8m0_scale>
__global__ void __launch_bounds__(512)
    quantize_1x128_kernel(const T *const input,
                          __nv_fp8_e4m3 *const output,
                          __nv_fp8_e4m3 *const output_transposed,
                          ScaleT *const scale,
                          ScaleT *const scale_transposed,
                          const size_t rows,
                          const size_t cols,
                          const size_t quanted_rows,
                          const size_t quanted_cols,
                          const float epsilon) {
  __shared__ __nv_fp8_e4m3 shm[128][129];
  __shared__ float block_scale[128];

  const size_t col_blocks = rows / k_block_span;  // rows is actually cols
  const size_t block_x = blockIdx.x % col_blocks;
  const size_t block_y = blockIdx.x / col_blocks;
  const size_t block_offset_x = block_x * k_block_span;
  const size_t block_offset_y = block_y * k_block_span;

  // 1. Load 128x128 block of input.
  v64_t<T> x[8];
  for (uint32_t i = 0; i < 8; i++) {
    size_t col_idx = block_offset_y + static_cast<size_t>(threadIdx.y) + i * 16;
    size_t row_idx = block_offset_x + static_cast<size_t>(threadIdx.x) * 4;
    size_t idx = col_idx * rows + row_idx;
    x[i] = *reinterpret_cast<const v64_t<T> *>(input + idx);
  }

  if constexpr (!return_transpose_only) {
    // 2. Compute scale along the row.
    ComputeRowScale<T, use_pow2_scale, using_ue8m0_scale>(
        x, block_scale, reinterpret_cast<T *>(shm), epsilon);

    // 3. Write 1x128 scale.
    if (threadIdx.y < 4) {
      size_t col_idx = block_offset_y + static_cast<size_t>(threadIdx.y) * 32 +
                       static_cast<size_t>(threadIdx.x);
      size_t row_idx = block_x;
      // block_scale stores multiplication scale now
      float store_scale = 1.0f / block_scale[threadIdx.y * 32 + threadIdx.x];

      size_t stride_rows = using_ue8m0_scale ? quanted_rows * 4 : quanted_rows;
      size_t idx;

      if (using_ue8m0_scale) {
        // 4. Whether it is 1*128 or 128*128, when output_scale_transpose is
        // True and using_ue8m0_scale is enabled, the calculation of idx needs
        // to be independently upgraded, because 4 original floats are packed as
        // a group and then transposed. If the transposed idx is calculated
        // directly, the data will be wrong.
        idx = output_scale_transpose
                  ? (row_idx / 4) * (cols * 4) + col_idx * 4 + (row_idx % 4)
                  : col_idx * stride_rows + row_idx;
      } else {
        idx = output_scale_transpose ? row_idx * stride_rows + col_idx
                                     : col_idx * stride_rows + row_idx;
      }
      StoreScale<ScaleT, using_ue8m0_scale>(scale, idx, store_scale);
    }

    // 4. Do quantization on X and write 128x128 output.
    for (uint32_t i = 0; i < 8; i++) {
      size_t col_idx =
          block_offset_y + static_cast<size_t>(threadIdx.y) + i * 16;
      size_t row_idx = block_offset_x + static_cast<size_t>(threadIdx.x) * 4;
      size_t idx = col_idx * rows + row_idx;
      float scale_val = block_scale[i * 16 + threadIdx.y];
      fp8x4_t data;
      for (uint32_t j = 0; j < 4; j++) {
        float x_scaled = static_cast<float>(x[i].val[j]) * scale_val;
        data.val[j] = static_cast<__nv_fp8_e4m3>(x_scaled);
      }
      *reinterpret_cast<fp8x4_t *>(output + idx) = data;
    }
  }

  if constexpr (input_transpose) {
    // 5. Compute scale along the column.
    ComputeColumnScale<T, use_pow2_scale, using_ue8m0_scale>(
        x, block_scale, reinterpret_cast<T *>(shm), epsilon);

    // 6. Write 1x128 transposed scale.
    if (threadIdx.y < 4) {
      size_t col_idx = block_y;
      size_t row_idx = block_offset_x + static_cast<size_t>(threadIdx.y) * 32 +
                       static_cast<size_t>(threadIdx.x);
      // block_scale stores multiplication scale
      float store_scale = 1.0f / block_scale[threadIdx.y * 32 + threadIdx.x];

      size_t stride_cols = using_ue8m0_scale ? quanted_cols * 4 : quanted_cols;
      size_t idx;
      if (using_ue8m0_scale) {
        // 4. Whether it is 1*128 or 128*128, when output_scale_transpose is
        // True and using_ue8m0_scale is enabled, the calculation of idx needs
        // to be independently upgraded, because 4 original floats are packed as
        // a group and then transposed. If the transposed idx is calculated
        // directly, the data will be wrong.
        idx = output_scale_transpose
                  ? (col_idx / 4) * (rows * 4) + row_idx * 4 + (col_idx % 4)
                  : row_idx * stride_cols + col_idx;
      } else {
        idx = output_scale_transpose ? col_idx * stride_cols + row_idx
                                     : row_idx * stride_cols + col_idx;
      }
      StoreScale<ScaleT, using_ue8m0_scale>(scale_transposed, idx, store_scale);
    }

    // 7. Do quantization on X and transpose in the block.
    for (uint32_t i = 0; i < 8; i++) {
      for (uint32_t j = 0; j < 4; j++) {
        float scale_val = block_scale[threadIdx.x * 4 + j];
        float x_scaled = static_cast<float>(x[i].val[j]) * scale_val;
        shm[threadIdx.x * 4 + j][i * 16 + threadIdx.y] =
            static_cast<__nv_fp8_e4m3>(x_scaled);
      }
    }
    __syncthreads();

    // 8. Write 128x128 transposed output.
    for (uint32_t i = 0; i < 8; i++) {
      size_t row_idx =
          block_offset_x + static_cast<size_t>(threadIdx.y) + i * 16;
      size_t col_idx = block_offset_y + static_cast<size_t>(threadIdx.x) * 4;
      size_t idx = row_idx * cols + col_idx;
      fp8x4_t data;
      for (uint32_t j = 0; j < 4; j++) {
        data.val[j] = shm[i * 16 + threadIdx.y][threadIdx.x * 4 + j];
      }
      *reinterpret_cast<fp8x4_t *>(output_transposed + idx) = data;
    }
  }
}

template <typename T,
          typename ScaleT,
          bool output_scale_transpose,
          bool use_pow2_scale,
          bool using_ue8m0_scale>
__global__ void quant_per_token_per_block(const T *input,
                                          __nv_fp8_e4m3 *quanted_res,
                                          ScaleT *quanted_scale,
                                          const int rows,
                                          const int cols,
                                          const int padded_rows,
                                          const float epsilon) {
  const size_t bid = blockIdx.x;
  const size_t tid = threadIdx.x;
  const size_t warp_id = tid / 32;
  const size_t lane_id = tid % 32;
  const size_t num_warp = blockDim.x / 32;
  static constexpr int NUM_PER_THREADS = 128 / 32;  // 4
  const size_t end_iter = (cols + 127) / 128;       // warp_iter_num

  AlignedVector<T, NUM_PER_THREADS> load_vec;
  AlignedVector<float, NUM_PER_THREADS> load_vec_float;
  AlignedVector<__nv_fp8_e4m3, NUM_PER_THREADS> res_vec;

  for (size_t token_idx = bid; token_idx < rows; token_idx += gridDim.x) {
    const T *input_now = input + token_idx * cols;
    __nv_fp8_e4m3 *quanted_res_now = quanted_res + token_idx * cols;
    // deal a block per warp
    for (size_t iter = warp_id; iter < end_iter; iter += num_warp) {
      size_t idx;
      if constexpr (using_ue8m0_scale) {
        idx = output_scale_transpose
                  ? (iter / 4) * padded_rows * 4 + token_idx * 4 + (iter % 4)
                  : token_idx * padded_rows * 4 + iter;
      } else {
        idx = output_scale_transpose ? iter * padded_rows + token_idx
                                     : token_idx * end_iter + iter;
      }
      const size_t offset = iter * 128 + lane_id * NUM_PER_THREADS;

      const bool is_valid_data = offset < cols;

      if (is_valid_data) {
        Load<T, NUM_PER_THREADS>(input_now + offset, &load_vec);
      } else {
#pragma unroll
        for (int vid = 0; vid < NUM_PER_THREADS; vid++) load_vec[vid] = T(0.f);
      }

      // get max value per thread
      float max_value_thread = -5e4;
#pragma unroll
      for (int vid = 0; vid < NUM_PER_THREADS; vid++) {
        load_vec_float[vid] = static_cast<float>(load_vec[vid]);
        max_value_thread = fmax(fabs(load_vec_float[vid]), max_value_thread);
      }

#pragma unroll
      for (uint32_t offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, max_value_thread, offset);
        max_value_thread = fmax(max_value_thread, other);
      }
      max_value_thread = __shfl_sync(0xFFFFFFFF, max_value_thread, 0);

      float blockwise_scale = ScaleWrapper < use_pow2_scale ||
                              using_ue8m0_scale > (max_value_thread, epsilon);
      float scale_to_store = 1.0f / blockwise_scale;
      // quant
#pragma unroll
      for (int vid = 0; vid < NUM_PER_THREADS; vid++) {
        res_vec[vid] =
            static_cast<__nv_fp8_e4m3>(load_vec_float[vid] * blockwise_scale);
      }
      // store
      if (is_valid_data) {
        Store<__nv_fp8_e4m3, NUM_PER_THREADS>(res_vec,
                                              quanted_res_now + offset);
      }

      if (lane_id == 0) {
        StoreScale<ScaleT, using_ue8m0_scale>(
            quanted_scale, idx, scale_to_store);
      }
    }
  }
}

template <typename T,
          typename ScaleT,
          bool output_scale_transpose,
          bool use_pow2_scale,
          bool using_ue8m0_scale>
__global__ void quant_per_token_per_block_padding(
    const T *const input,
    __nv_fp8_e4m3 *const quanted_res,
    ScaleT *const quanted_scale,
    const size_t rows,
    const size_t cols,
    const size_t padded_rows,
    const float epsilon) {
  const size_t bid = blockIdx.x;
  const size_t tid = threadIdx.x;
  const size_t warp_id = tid / 32;
  const size_t lane_id = tid % 32;
  const size_t num_warp = blockDim.x / 32;
  static constexpr int NUM_PER_THREADS = 128 / 32;  // 4
  const size_t end_iter = cols / 128;               // warp_iter_num

  AlignedVector<T, NUM_PER_THREADS> load_vec;
  AlignedVector<float, NUM_PER_THREADS> load_vec_float;
  AlignedVector<__nv_fp8_e4m3, NUM_PER_THREADS> res_vec;

  for (size_t token_idx = bid; token_idx < rows; token_idx += gridDim.x) {
    const T *input_now = input + token_idx * cols;
    __nv_fp8_e4m3 *quanted_res_now = quanted_res + token_idx * cols;
    // deal a block per warp
    for (size_t iter = warp_id; iter < end_iter; iter += num_warp) {
      size_t idx;
      if constexpr (using_ue8m0_scale) {
        idx = output_scale_transpose
                  ? (iter / 4) * padded_rows * 4 + token_idx * 4 + (iter % 4)
                  : token_idx * padded_rows * 4 + iter;
      } else {
        idx = output_scale_transpose ? iter * padded_rows + token_idx
                                     : token_idx * end_iter + iter;
      }
      const size_t offset = iter * 128 + lane_id * NUM_PER_THREADS;

      Load<T, NUM_PER_THREADS>(input_now + offset, &load_vec);

      // get max value per thread
      float max_value_thread = -5e4;
#pragma unroll
      for (int vid = 0; vid < NUM_PER_THREADS; vid++) {
        load_vec_float[vid] = static_cast<float>(load_vec[vid]);
        max_value_thread = fmax(fabs(load_vec_float[vid]), max_value_thread);
      }

#pragma unroll
      for (uint32_t offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, max_value_thread, offset);
        max_value_thread = fmax(max_value_thread, other);
      }
      max_value_thread = __shfl_sync(0xFFFFFFFF, max_value_thread, 0);

      float blockwise_scale = ScaleWrapper < use_pow2_scale ||
                              using_ue8m0_scale > (max_value_thread, epsilon);
      float scale_to_store = 1.0f / blockwise_scale;
      // quant
#pragma unroll
      for (int vid = 0; vid < NUM_PER_THREADS; vid++) {
        res_vec[vid] =
            static_cast<__nv_fp8_e4m3>(load_vec_float[vid] * blockwise_scale);
      }
      // store
      Store<__nv_fp8_e4m3, NUM_PER_THREADS>(res_vec, quanted_res_now + offset);
      if (lane_id == 0) {
        StoreScale<ScaleT, using_ue8m0_scale>(
            quanted_scale, idx, scale_to_store);
      }
    }
  }
}

template <typename Context,
          typename T,
          bool using_1x128_vec_quant,
          bool input_transpose,
          bool output_scale_transpose,
          bool return_transpose_only,
          bool using_pow2_scale,
          bool using_ue8m0_scale>
void FP8QuantBlockWiseKernelImpl(const Context &dev_ctx,
                                 const DenseTensor &X,
                                 float epsilon,
                                 DenseTensor *out,
                                 DenseTensor *scale,
                                 DenseTensor *out_transposed,
                                 DenseTensor *scale_transposed) {
  const size_t src_rows = X.dims()[0];
  const size_t src_cols = X.dims()[1];

  const size_t quanted_cols = scale->dims()[1];
  const size_t quanted_rows = scale_transposed->dims()[1];

  using NvType = typename CudaTypeTraits<T>::Type;
  using ScaleT = typename std::conditional_t<using_ue8m0_scale, int, float>;

  if (src_cols % 128 != 0) {
    PD_CHECK(
        !input_transpose && !return_transpose_only,
        "When cols is not aligned to 128, only supports: input_transpose=False "
        "and return_transpose_only=False.");
    PD_CHECK(using_1x128_vec_quant,
             "When cols is not aligned to 128, only supports: "
             "using_1x128_vec_quant.");
    PD_CHECK(src_cols % 4 == 0, "src_cols must be divisible by 4.");

    const int device_id = phi::backends::gpu::GetCurrentDeviceId();
    const int sm_count = phi::backends::gpu::GetGPUMultiProcessors(device_id);
    const size_t min_grid_x = sm_count * 8;
    const size_t min_block_x = 1024;
    const size_t gridx = std::min(min_grid_x, src_rows);
    const size_t blockx = std::min(min_block_x, (src_cols + 127) / 128 * 32);

    quant_per_token_per_block<NvType,
                              ScaleT,
                              output_scale_transpose,
                              using_pow2_scale,
                              using_ue8m0_scale>
        <<<gridx, blockx, 0, dev_ctx.stream()>>>(
            reinterpret_cast<const NvType *>(X.data<T>()),
            reinterpret_cast<__nv_fp8_e4m3 *>(out->data<phi::float8_e4m3fn>()),
            reinterpret_cast<ScaleT *>(scale->data<ScaleT>()),
            src_rows,
            src_cols,
            quanted_cols,
            epsilon);
  } else if (src_rows % 128 != 0) {
    PD_CHECK(
        !input_transpose && !return_transpose_only,
        "When rows is not aligned to 128, only supports: input_transpose=False "
        "and return_transpose_only=False.");
    PD_CHECK(using_1x128_vec_quant,
             "When rows is not aligned to 128, only supports: "
             "using_1x128_vec_quant.");
    PD_CHECK(src_cols % 128 == 0, "src_cols must be divisible by 128.");

    const int device_id = phi::backends::gpu::GetCurrentDeviceId();
    const int sm_count = phi::backends::gpu::GetGPUMultiProcessors(device_id);
    const size_t min_grid_x = sm_count * 8;
    const size_t min_block_x = 1024;
    const size_t gridx = std::min(min_grid_x, src_rows);
    const size_t blockx = std::min(min_block_x, src_cols / 128 * 32);

    quant_per_token_per_block_padding<NvType,
                                      ScaleT,
                                      output_scale_transpose,
                                      using_pow2_scale,
                                      using_ue8m0_scale>
        <<<gridx, blockx, 0, dev_ctx.stream()>>>(
            reinterpret_cast<const NvType *>(X.data<T>()),
            reinterpret_cast<__nv_fp8_e4m3 *>(out->data<phi::float8_e4m3fn>()),
            reinterpret_cast<ScaleT *>(scale->data<ScaleT>()),
            src_rows,
            src_cols,
            quanted_cols,
            epsilon);
  } else {
    dim3 block, grid;
    const size_t blocks_per_row = src_cols / k_block_span;
    if constexpr (using_1x128_vec_quant) {
      block.x = 32;
      block.y = 16;
    } else {
      block.x = 256;
    }
    grid.x = (src_cols / k_block_span) * (src_rows / k_block_span);

    auto kernel = using_1x128_vec_quant
                      ? quantize_1x128_kernel<NvType,
                                              ScaleT,
                                              input_transpose,
                                              output_scale_transpose,
                                              return_transpose_only,
                                              using_pow2_scale,
                                              using_ue8m0_scale>
                      : quantize_128x128_kernel<NvType,
                                                ScaleT,
                                                input_transpose,
                                                output_scale_transpose,
                                                return_transpose_only,
                                                using_pow2_scale,
                                                using_ue8m0_scale>;

    kernel<<<grid, block, 0, dev_ctx.stream()>>>(
        reinterpret_cast<const NvType *>(X.data<T>()),
        reinterpret_cast<__nv_fp8_e4m3 *>(out->data<phi::float8_e4m3fn>()),
        input_transpose ? reinterpret_cast<__nv_fp8_e4m3 *>(
                              out_transposed->data<phi::float8_e4m3fn>())
                        : nullptr,
        reinterpret_cast<ScaleT *>(scale->data<ScaleT>()),
        input_transpose
            ? reinterpret_cast<ScaleT *>(scale_transposed->data<ScaleT>())
            : nullptr,
        src_cols,
        src_rows,
        quanted_cols,
        quanted_rows,
        epsilon);
  }
}

// T is x's input type and out_dtype is in args
template <typename T, typename Context>
void FP8QuantBlockWiseKernel(const Context &dev_ctx,
                             const DenseTensor &X,
                             float epsilon,
                             bool using_1x128_vec_quant,
                             bool input_transpose,
                             bool output_scale_transpose,
                             bool return_transpose_only,
                             bool using_e5m2,
                             bool using_pow2_scale,
                             bool using_ue8m0_scale,
                             DenseTensor *out,
                             DenseTensor *scale,
                             DenseTensor *out_transposed,
                             DenseTensor *scale_transposed) {
  PD_CHECK(X.dtype() == phi::DataType::BFLOAT16 ||
               X.dtype() == phi::DataType::FLOAT16,
           "X datatype error, can only be bfloat16 or float16");

  dev_ctx.template Alloc<phi::float8_e4m3fn>(out);
  if (using_ue8m0_scale) {
    dev_ctx.template Alloc<int32_t>(scale);
  } else {
    dev_ctx.template Alloc<float>(scale);
  }

  if (input_transpose) {
    dev_ctx.template Alloc<phi::float8_e4m3fn>(out_transposed);
    if (using_ue8m0_scale) {
      dev_ctx.template Alloc<int32_t>(scale_transposed);
    } else {
      dev_ctx.template Alloc<float>(scale_transposed);
    }
  }

  if (X.numel() == 0) {
    return;
  }

  // Currently we only support bfloat16 as input type,
  // fp8_e4m3fn as output type.
  DISPATCH_BOOL(
      using_1x128_vec_quant,
      k_using_1x128_vec_quant,
      DISPATCH_BOOL(
          input_transpose,
          k_input_transpose,
          DISPATCH_BOOL(
              output_scale_transpose,
              k_output_scale_transpose,
              DISPATCH_BOOL(
                  return_transpose_only,
                  k_return_transpose_only,
                  DISPATCH_BOOL(
                      using_pow2_scale,
                      k_using_pow2_scale,
                      DISPATCH_BOOL(
                          using_ue8m0_scale,
                          k_using_ue8m0_scale,
                          FP8QuantBlockWiseKernelImpl<Context,
                                                      T,
                                                      k_using_1x128_vec_quant,
                                                      k_input_transpose,
                                                      k_output_scale_transpose,
                                                      k_return_transpose_only,
                                                      k_using_pow2_scale,
                                                      k_using_ue8m0_scale>(
                              dev_ctx,
                              X,
                              epsilon,
                              out,
                              scale,
                              out_transposed,
                              scale_transposed);))))));
}
}  // namespace phi

PD_REGISTER_KERNEL(fp8_quant_blockwise,
                   GPU,
                   ALL_LAYOUT,
                   phi::FP8QuantBlockWiseKernel,
                   phi::float16,
                   phi::bfloat16,
                   float,
                   double) {}

// NOLINTEND
