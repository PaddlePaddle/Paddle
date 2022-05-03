/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace paddle {
namespace operators {

using framework::Tensor;

namespace plat = paddle::platform;

#define FINAL_MASK 0xffffffff
#define DIVUP(x, y) (((x) + (y)-1) / (y))
#define ROUNDUP(x, y) (DIVUP((x), (y)) * (y))

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    return val;
}

template<typename T>
__inline__ __device__ T warpReduceMax(T val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

template <typename T, int VEC_SIZE, int ELEMENTS_PER_THREADS>
__global__ void fused_softmax_mask_vec_kernel(
    T* dst, const T* src, const bool* mask, int seq_len) {
  constexpr int block_size = 128;
  constexpr int warp_size = 32;
  constexpr int warps_per_block = block_size / warp_size;

  // blockDim/threadIdx = (warp_size, warps_per_block)
  // gridDim/blockIdx = (DIVUP(seq_len, warps_per_block), batch_size, head_num)

  // every block processes 4(warps_per_block) sequences
  // seq_id = seq_id * 4 + warp_id, eg.seq_len=128, 127=31*4+3
  int seq_id = blockIdx.x * warps_per_block + threadIdx.y;
  if (seq_id >= seq_len) return;

  // ((bid*head_num + hid)*seq_len + seq_id) * seq_len
  int offset = ((blockIdx.y * gridDim.z + blockIdx.z) * seq_len + seq_id) * seq_len;
  // (bid * seq_len + seq_id) * seq_len
  int mask_offset = (blockIdx.y * seq_len + seq_id) * seq_len;
  src += offset;
  dst += offset;
  mask += mask_offset;

  static_assert(ELEMENTS_PER_THREADS % VEC_SIZE == 0, "");
  constexpr int VEC_NUMS = ELEMENTS_PER_THREADS / VEC_SIZE;
  using VecT = phi::AlignedVector<T, VEC_SIZE>;
  using VecBool = phi::AlignedVector<bool, VEC_SIZE>;

  VecT elements[VEC_NUMS];
  VecBool tmp_mask;
  float max_val = -std::numeric_limits<float>::infinity();

  for (int i = 0; (i * warp_size + threadIdx.x) * VEC_SIZE < seq_len; ++i) {
    phi::Load(src + (i * warp_size + threadIdx.x) * VEC_SIZE, &elements[i]);
    phi::Load(mask + (i * warp_size + threadIdx.x) * VEC_SIZE, &tmp_mask);
    #pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      elements[i][j] = elements[i][j] +
                       (tmp_mask[j] ? static_cast<T>(0) :
                                    static_cast<T>(-10000));
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

#define SOFTMAX_MASK_KERNEL(T, VEC_SIZE, ELEMENTS) \
  fused_softmax_mask_vec_kernel<T, VEC_SIZE, ELEMENTS><<<grid, block, 0, stream>>>( \
      dst, src, mask, seq_len)

//#define SELECT_SOFTMAX_MASK_KERNEL(T, ELEMENTS) \
//    do { \
//      if (sizeof(T) == 2 && seq_len % 8 == 0) { \
//        fused_softmax_mask_vec_kernel<plat::float16, 8, ELEMENTS><<<grid, block, 0, stream>>>( \
//            (plat::float16*)dst, (const plat::float16*)src, mask, seq_len); \
//      } \
//      else if (seq_len % 4 == 0) SOFTMAX_MASK_KERNEL(T, 4, ELEMENTS); \
//      else if (seq_len % 2 == 0) SOFTMAX_MASK_KERNEL(T, 2, ELEMENTS); \
//      else SOFTMAX_MASK_KERNEL(T, 1, ELEMENTS);   \
//    } while(0)

// FIXME(wangxi): It is found that the performance of VEC_SIZE=2 is better
//  than that of =4 and =8. Further analysis of the kernel is needed later.
#define SELECT_SOFTMAX_MASK_KERNEL(T, ELEMENTS) \
    do { \
      if (seq_len % 2 == 0) SOFTMAX_MASK_KERNEL(T, 2, ELEMENTS); \
      else SOFTMAX_MASK_KERNEL(T, 1, ELEMENTS);   \
    } while(0)

template <typename T>
void fused_softmax_mask(const T* src,
                        const bool* mask,
                        T* dst,
                        const int batch_size,
                        const int head_num,
                        const int seq_len,
                        cudaStream_t stream) {
  assert(seq_len > 0 && seq_len <= 4096);

  constexpr int block_size = 128;
  constexpr int warp_size = 32;
  constexpr int warps_per_block = block_size / warp_size;

  // put head_num to the outside for mask
  dim3 block(warp_size, warps_per_block);
  dim3 grid(DIVUP(seq_len, warps_per_block), batch_size, head_num);

  if (seq_len <= 32) {
    SOFTMAX_MASK_KERNEL(T, 1, 1);
  } else if (seq_len <= 64) {
    // if (seq_len % 2 == 0) SOFTMAX_MASK_KERNEL(T, 2, 2);
    // else SOFTMAX_MASK_KERNEL(T, 1, 2);
    SELECT_SOFTMAX_MASK_KERNEL(T, 2);
  } else if (seq_len <= 128) {
    // if (seq_len % 4 == 0) SOFTMAX_MASK_KERNEL(T, 4, 4);
    // else if (seq_len % 2 == 0) SOFTMAX_MASK_KERNEL(T, 2, 4);
    // else SOFTMAX_MASK_KERNEL(T, 1, 4);
    SELECT_SOFTMAX_MASK_KERNEL(T, 4);
  } else if (seq_len <= 256) {
    SELECT_SOFTMAX_MASK_KERNEL(T, 8);
  } else if (seq_len <= 512) {
    SELECT_SOFTMAX_MASK_KERNEL(T, 16);
  } else if (seq_len <= 1024) {
    SELECT_SOFTMAX_MASK_KERNEL(T, 32);
  } else if (seq_len <= 2048) {
    SELECT_SOFTMAX_MASK_KERNEL(T, 64);
  } else if (seq_len <= 4096) {
    SELECT_SOFTMAX_MASK_KERNEL(T, 128);
  }
}

#undef SELECT_SOFTMAX_MASK_KERNEL
#undef SOFTMAX_MASK_KERNEL
#undef ROUNDUP
#undef DIVUP

}  // namespace operators
}  // namespace paddle
