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

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <curand_kernel.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>
#endif

#include "paddle/phi/kernels/funcs/aligned_vector.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#ifdef PADDLE_WITH_HIP
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

#define MASK 0xffffffff

namespace phi {
namespace fusion {

__device__ __inline__ void load_data(dtype::float16* dst,
                                     const dtype::float16* src) {
  *(reinterpret_cast<float2*>(dst)) = *(reinterpret_cast<const float2*>(src));
}

__device__ __inline__ void load_data(float* dst, const float* src) {
  *(reinterpret_cast<float4*>(dst)) = *(reinterpret_cast<const float4*>(src));
}

inline int get_pow2(int value) {
  // get next pow2 index
  int pow2_index = 0;
  while ((1 << pow2_index) < value) {
    ++pow2_index;
  }
  return pow2_index;
}

template <typename T>
struct AddOP {
  __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct MaxOP {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename T>
__device__ __forceinline__ T
warp_shfl_xor(T value, int laneMask, int width, unsigned int mask = MASK) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}

template <typename T, int batch, int width, template <typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(T* sum) {
  ReduceOp<T> r;
#pragma unroll
  for (int offset = width / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < batch; ++i) {
      T b = warp_shfl_xor(sum[i], offset, width);
      sum[i] = r(sum[i], b);
    }
  }
}

#if defined(PADDLE_WITH_CUDA)

#define FINAL_MASK 0xffffffff
#define DIV_UP(x, y) (((x) + (y)-1) / (y))

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

inline int ElementsCeil(int seq_len) {
  int elements = 1;
  while (elements * 32 < seq_len) elements *= 2;
  return elements;
}

template <typename T, int VEC_SIZE, int ELEMENTS_PER_THREADS>
__global__ void FusedSoftmaxMaskVecKernel(T* dst,
                                          const T* src,
                                          const T* mask,
                                          int seq_len) {
  constexpr int block_size = 128;
  constexpr int warp_size = 32;
  constexpr int warps_per_block = block_size / warp_size;

  // blockDim/threadIdx = (warp_size, warps_per_block)
  // gridDim/blockIdx = (DIV_UP(seq_len, warps_per_block), batch_size, head_num)
  // every block processes 4(warps_per_block) sequences
  // seq_id = seq_id * 4 + warp_id, eg.seq_len=128, 127=31*4+3
  int seq_id = blockIdx.x * warps_per_block + threadIdx.y;
  if (seq_id >= seq_len) return;

  // ((bid*head_num + hid)*seq_len + seq_id) * seq_len
  int offset =
      ((blockIdx.y * gridDim.z + blockIdx.z) * seq_len + seq_id) * seq_len;
  // (bid * seq_len + seq_id) * seq_len
  int mask_offset = (blockIdx.y * seq_len + seq_id) * seq_len;
  src += offset;
  dst += offset;
  mask += mask_offset;

  static_assert(ELEMENTS_PER_THREADS % VEC_SIZE == 0, "");
  constexpr int VEC_NUMS = ELEMENTS_PER_THREADS / VEC_SIZE;
  using VecT = phi::AlignedVector<T, VEC_SIZE>;

  VecT elements[VEC_NUMS];
  VecT tmp_mask;
  float max_val = -std::numeric_limits<float>::infinity();

  for (int i = 0; (i * warp_size + threadIdx.x) * VEC_SIZE < seq_len; ++i) {
    phi::Load(src + (i * warp_size + threadIdx.x) * VEC_SIZE, &elements[i]);
    phi::Load(mask + (i * warp_size + threadIdx.x) * VEC_SIZE, &tmp_mask);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      // TODO(wangxi): vec add
      elements[i][j] += tmp_mask[j];
      max_val = max(max_val, static_cast<float>(elements[i][j]));
    }
  }
  max_val = warpReduceMax(max_val);

  float sum_val = 0;
  for (int i = 0; (i * warp_size + threadIdx.x) * VEC_SIZE < seq_len; ++i) {
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      float tmp = __expf(static_cast<float>(elements[i][j]) - max_val);
      sum_val += tmp;
      elements[i][j] = static_cast<T>(tmp);
    }
  }
  sum_val = warpReduceSum(sum_val);
  float mean_val = __fdividef(1.0f, sum_val + 1e-6f);

  for (int i = 0; (i * warp_size + threadIdx.x) * VEC_SIZE < seq_len; ++i) {
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      float tmp = static_cast<float>(elements[i][j]) * mean_val;
      elements[i][j] = static_cast<T>(tmp);
    }
    phi::Store(elements[i], dst + (i * warp_size + threadIdx.x) * VEC_SIZE);
  }
}

#define SOFTMAX_MASK_KERNEL(VEC_SIZE, ELEMENTS)    \
  FusedSoftmaxMaskVecKernel<T, VEC_SIZE, ELEMENTS> \
      <<<grid, block, 0, stream>>>(dst, src, mask, seq_len);

#define SELECT_SOFTMAX_MASK_KERNEL(ELEMENTS) \
  do {                                       \
    if (seq_len % 2 == 0) {                  \
      SOFTMAX_MASK_KERNEL(2, ELEMENTS);      \
    } else {                                 \
      SOFTMAX_MASK_KERNEL(1, ELEMENTS);      \
    }                                        \
  } while (0)

#define CASE_SOFTMAX_MASK_KERNEL(ELEMENTS) \
  case ELEMENTS: {                         \
    SELECT_SOFTMAX_MASK_KERNEL(ELEMENTS);  \
    break;                                 \
  }

// template <typename T, typename MaskT = T>
template <typename T>
void LaunchFusedSoftmaxMaskKernel(const T* src,
                                  const T* mask,
                                  T* dst,
                                  const int batch_size,
                                  const int head_num,
                                  const int seq_len,
                                  cudaStream_t stream) {
  PADDLE_ENFORCE_EQ(seq_len > 0 && seq_len <= 4096,
                    true,
                    errors::InvalidArgument("seq_len must be between (0, 4096] "
                                            "received the seq_len is %d",
                                            seq_len));

  constexpr int block_size = 128;
  constexpr int warp_size = 32;
  constexpr int warps_per_block = block_size / warp_size;

  // put head_num to the outside for mask
  dim3 block(warp_size, warps_per_block);
  dim3 grid(DIV_UP(seq_len, warps_per_block), batch_size, head_num);

  int elements = ElementsCeil(seq_len);
  switch (elements) {
    case 1: {  // <=32
      SOFTMAX_MASK_KERNEL(1, 1);
      break;
    }
    case 2: {  // <=64
      // if (seq_len % 2 == 0) SOFTMAX_MASK_KERNEL(2, 2);
      // else SOFTMAX_MASK_KERNEL(1, 2);
      SELECT_SOFTMAX_MASK_KERNEL(2);
      break;
    }
    case 4: {  // <=128
      // if (seq_len % 4 == 0) SOFTMAX_MASK_KERNEL(4, 4);
      // else if (seq_len % 2 == 0) SOFTMAX_MASK_KERNEL(2, 4);
      // else SOFTMAX_MASK_KERNEL(1, 4);
      SELECT_SOFTMAX_MASK_KERNEL(4);
      break;
    }
      CASE_SOFTMAX_MASK_KERNEL(8);    // <=256
      CASE_SOFTMAX_MASK_KERNEL(16);   // <=512
      CASE_SOFTMAX_MASK_KERNEL(32);   // <=1024
      CASE_SOFTMAX_MASK_KERNEL(64);   // <=2048
      CASE_SOFTMAX_MASK_KERNEL(128);  // <=4096
    default:
      PADDLE_THROW(errors::InvalidArgument(
          "seq_len must be between (0, 4096], received the seq_len is %d",
          seq_len));
  }
}
#endif

}  // namespace fusion
}  // namespace phi

#endif
