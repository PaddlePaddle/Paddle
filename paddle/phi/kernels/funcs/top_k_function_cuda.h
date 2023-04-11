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
#include <stdio.h>

#include <cstdio>
#include <vector>
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
#endif
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"

#define FINAL_MASK 0xffffffff
#define WARP_SIZE 32
#define MAX_NUM_THREADS 1024

inline static size_t divide_round_up(size_t n, size_t q) {
  return n % q == 0 ? n / q : n / q + 1;
}

inline static size_t round_up(size_t n, size_t q) {
  return divide_round_up(n, q) * q;
}

#ifdef __HIPCC__
namespace rocprim {
namespace detail {
template <>
struct radix_key_codec_base<phi::dtype::float16>
    : radix_key_codec_integral<phi::dtype::float16, uint16_t> {};

template <>
struct radix_key_codec_base<phi::dtype::bfloat16>
    : radix_key_codec_integral<phi::dtype::bfloat16, uint16_t> {};
}  // namespace detail
}  // namespace rocprim
namespace cub = hipcub;
#else
// set cub base traits in order to handle float16
namespace cub {
template <>
struct NumericTraits<phi::dtype::float16>
    : BaseTraits<FLOATING_POINT, true, false, uint16_t, phi::dtype::float16> {};

template <>
struct NumericTraits<phi::dtype::bfloat16>
    : BaseTraits<FLOATING_POINT, true, false, uint16_t, phi::dtype::bfloat16> {
};

}  // namespace cub
#endif

namespace phi {
namespace funcs {

using Tensor = phi::DenseTensor;

inline void GetDims(
    const phi::DDim& dim, int axis, int* pre, int* n, int* post) {
  *pre = 1;
  *post = 1;
  *n = dim[axis];
  for (int i = 0; i < axis; ++i) {
    (*pre) *= dim[i];
  }
  for (int i = axis + 1; i < dim.size(); ++i) {
    (*post) *= dim[i];
  }
}

struct SegmentOffsetIter {
  EIGEN_DEVICE_FUNC
  explicit SegmentOffsetIter(int num_cols) : num_cols_(num_cols) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int operator()(int idx) const {
    return idx * num_cols_;
  }

  int num_cols_;
};

// Iter using into a column
struct ColumnIndexIter {
  explicit ColumnIndexIter(int num_cols) : num_cols_(num_cols) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int operator()(
      const Eigen::array<int, 1>& ix) const {
    return ix[0] % num_cols_;
  }

  int num_cols_;
};

inline static int GetDesiredBlockDim(int dim) {
  if (dim > 128) {
    return 256;
  } else if (dim > 64) {
    return 128;
  } else if (dim > 32) {
    return 64;
  } else {
    return 32;
  }
}

inline static int getMaxLength(int k) {
  if (k / 5 < 1) {
    return 1;
  } else if (k / 5 >= 1) {
    return min(k / 5, 5);
  }
}

template <typename T>
__global__ void InitIndex(T* indices, T num_rows, T num_cols) {
  int col_id = threadIdx.x;
  int row_id = blockIdx.x;

  for (int64_t j = row_id; j < num_rows; j += gridDim.x) {
    for (int64_t i = col_id; i < num_cols; i += blockDim.x) {
      indices[j * num_cols + i] = i;
    }
  }
}

template <typename T>
struct Pair {
  __device__ __forceinline__ Pair() {}
  __device__ __forceinline__ Pair(T value, int64_t id) : v(value), id(id) {}

  __device__ __forceinline__ void set(T value, int64_t id) {
    v = value;
    id = id;
  }

  __device__ __forceinline__ void operator=(const Pair<T>& in) {
    v = in.v;
    id = in.id;
  }

  __device__ __forceinline__ bool operator<(const T value) const {
    return (v < value);
  }

  __device__ __forceinline__ bool operator>(const T value) const {
    return (v > value);
  }
  __device__ __forceinline__ bool operator<(const Pair<T>& in) const {
    return (v < in.v) || ((v == in.v) && (id > in.id));
  }

  __device__ __forceinline__ bool operator>(const Pair<T>& in) const {
    return (v > in.v) || ((v == in.v) && (id < in.id));
  }

  T v;
  int64_t id;
};

template <typename T>
__device__ __forceinline__ void AddTo(Pair<T> topk[],
                                      const Pair<T>& p,
                                      int beam_size,
                                      const bool& largest) {
  for (int k = beam_size - 2; k >= 0; k--) {
    if (largest) {
      if (topk[k] < p) {
        topk[k + 1] = topk[k];
      } else {
        topk[k + 1] = p;
        return;
      }
    } else {
      if (topk[k] > p) {
        topk[k + 1] = topk[k];
      } else {
        topk[k + 1] = p;
        return;
      }
    }
  }
  topk[0] = p;
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[],
                                        const T* src,
                                        int idx,
                                        int dim,
                                        int beam_size,
                                        const bool& largest) {
  while (idx < dim) {
    if (largest) {
      if (topk[beam_size - 1] < src[idx]) {
        Pair<T> tmp(src[idx], idx);
        AddTo<T>(topk, tmp, beam_size, largest);
      }
    } else {
      if (topk[beam_size - 1] > src[idx]) {
        Pair<T> tmp(src[idx], idx);
        AddTo<T>(topk, tmp, beam_size, largest);
      }
    }
    idx += BlockSize;
  }
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[],
                                        const T* src,
                                        int idx,
                                        int dim,
                                        const Pair<T>& max,
                                        int beam_size,
                                        const bool& largest) {
  while (idx < dim) {
    if (largest) {
      if (topk[beam_size - 1] < src[idx]) {
        Pair<T> tmp(src[idx], idx);
        if (tmp < max) {
          AddTo<T>(topk, tmp, beam_size, largest);
        }
      }
    } else {
      if (topk[beam_size - 1] > src[idx]) {
        Pair<T> tmp(src[idx], idx);
        if (tmp > max) {
          AddTo<T>(topk, tmp, beam_size, largest);
        }
      }
    }
    idx += BlockSize;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void ThreadGetTopK(Pair<T> topk[],
                                              int* beam,
                                              int beam_size,
                                              const T* src,
                                              bool* firstStep,
                                              bool* is_empty,
                                              Pair<T>* max,
                                              int dim,
                                              const int tid,
                                              bool largest) {
  if (*beam > 0) {
    int length = (*beam) < beam_size ? *beam : beam_size;
    if (*firstStep) {
      *firstStep = false;
      GetTopK<T, BlockSize>(topk, src, tid, dim, length, largest);
    } else {
      for (int k = 0; k < MaxLength; k++) {
        if (k < MaxLength - (*beam)) {
          topk[k] = topk[k + *beam];
        } else {
          if (largest) {
            topk[k].set(-static_cast<T>(INFINITY), -1);
          } else {
            topk[k].set(static_cast<T>(INFINITY), -1);
          }
        }
      }
      if (!(*is_empty)) {
        GetTopK<T, BlockSize>(
            topk + MaxLength - *beam, src, tid, dim, *max, length, largest);
      }
    }

    *max = topk[MaxLength - 1];
    if ((*max).id == -1) *is_empty = true;
    *beam = 0;
  }
}

template <typename T>
__forceinline__ __device__ Pair<T> WarpReduce(Pair<T> input,
                                              const bool& largest) {
  if (largest) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      T tmp_val =
          phi::backends::gpu::CudaShuffleDownSync(FINAL_MASK, input.v, offset);
      int tmp_id =
          phi::backends::gpu::CudaShuffleDownSync(FINAL_MASK, input.id, offset);
      if (input.v < tmp_val || (input.v == tmp_val && input.id > tmp_id)) {
        input.v = tmp_val;
        input.id = tmp_id;
      }
    }
  } else {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      T tmp_val =
          phi::backends::gpu::CudaShuffleDownSync(FINAL_MASK, input.v, offset);
      int tmp_id =
          phi::backends::gpu::CudaShuffleDownSync(FINAL_MASK, input.id, offset);
      if (input.v > tmp_val || (input.v == tmp_val && input.id > tmp_id)) {
        input.v = tmp_val;
        input.id = tmp_id;
      }
    }
  }
  return input;
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void BlockReduce(Pair<T> shared_max[],
                                            Pair<T> topk[],
                                            T** topVal,
                                            int64_t** topIds,
                                            int* beam,
                                            int* k,
                                            const int tid,
                                            const int wid,
                                            const int lane,
                                            const bool& largest) {
  while (true) {
    __syncthreads();
    Pair<T> input_now = topk[0];
    input_now = WarpReduce(input_now, largest);

    if (lane == 0) {
      shared_max[wid] = input_now;
    }
    __syncthreads();
    if (largest) {
      input_now = (tid < BlockSize / 32)
                      ? shared_max[lane]
                      : Pair<T>(-static_cast<T>(INFINITY), -1);
    } else {
      input_now = (tid < BlockSize / 32)
                      ? shared_max[lane]
                      : Pair<T>(static_cast<T>(INFINITY), -1);
    }
    if (wid == 0) {
      input_now = WarpReduce(input_now, largest);
      if (lane == 0) shared_max[0] = input_now;
    }
    __syncthreads();

    if (tid == 0) {
      **topVal = input_now.v;
      **topIds = input_now.id;
      (*topVal)++;
      (*topIds)++;
    }
    int tid_max = shared_max[0].id % BlockSize;
    if (tid == tid_max) {
      (*beam)++;
      if (*beam < MaxLength) {
        topk[0] = topk[*beam];
      }
    }
    if (--(*k) == 0) break;

    unsigned mask = 0u;
    CREATE_SHFL_MASK(mask, true);
    if (tid_max / 32 == wid) {
      if (phi::backends::gpu::CudaShuffleSync(mask, *beam, tid_max % 32, 32) ==
          MaxLength)
        break;
    }
  }
}

/**
 * Each block compute one sample.
 * In a block:
 * 1. every thread get top MaxLength value;
 * 2. merge to sh_topk, block reduce and get max value;
 * 3. go to the second setp, until one thread's topk value is null;
 * 4. go to the first setp, until get the topk value.
 */

template <typename T, int MaxLength, int BlockSize>
__global__ void KeMatrixTopK(T* output,
                             int output_stride,
                             int64_t* indices,
                             const T* src,
                             int lds,
                             int dim,
                             int k,
                             int grid_dim,
                             int num,
                             bool largest = true) {
  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lane = tid % 32;
  const int bid = blockIdx.x;
  for (int i = bid; i < num; i += grid_dim) {
    int top_num = k;
    __shared__ Pair<T> shared_max[BlockSize / 32];
    T* out = output + i * output_stride;
    int64_t* inds = indices + i * k;
    Pair<T> topk[MaxLength];
    int beam = MaxLength;
    Pair<T> max;
    bool is_empty = false;
    bool firststep = true;

    for (int j = 0; j < MaxLength; j++) {
      if (largest) {
        topk[j].set(-static_cast<T>(INFINITY), -1);
      } else {
        topk[j].set(static_cast<T>(INFINITY), -1);
      }
    }
    while (top_num) {
      ThreadGetTopK<T, MaxLength, BlockSize>(topk,
                                             &beam,
                                             k,
                                             src + i * lds,
                                             &firststep,
                                             &is_empty,
                                             &max,
                                             dim,
                                             tid,
                                             largest);
      BlockReduce<T, MaxLength, BlockSize>(shared_max,
                                           topk,
                                           &out,
                                           &inds,
                                           &beam,
                                           &top_num,
                                           tid,
                                           wid,
                                           lane,
                                           largest);
    }
  }
}

/*---------------------------Radix TopK Begin------------------*/
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 9000
constexpr int RADIX_BITS = 2;  // digits are base-(2 ^ RADIX_BITS)
constexpr int RADIX_SIZE = 4;  // 2 ^ RADIX_BITS
constexpr int RADIX_MASK = (RADIX_SIZE - 1);

/*---------------------------Helper Structs------------------*/
template <typename T>
struct Bitfield {};

template <>
struct Bitfield<unsigned int> {
  static __device__ __forceinline__ unsigned int GetBitfield(unsigned int val,
                                                             int pos,
                                                             int len) {
    unsigned int ret;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
    return ret;
  }

  static __device__ __forceinline__ unsigned int SetBitfield(
      unsigned int val, unsigned int to_insert, int pos, int len) {
    unsigned int ret;
    asm("bfi.b32 %0, %1, %2, %3, %4;"
        : "=r"(ret)
        : "r"(to_insert), "r"(val), "r"(pos), "r"(len));
    return ret;
  }
};

template <>
struct Bitfield<uint64_t> {
  static __device__ __forceinline__ uint64_t GetBitfield(uint64_t val,
                                                         int pos,
                                                         int len) {
    uint64_t ret;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
    return ret;
  }

  static __device__ __forceinline__ uint64_t SetBitfield(uint64_t val,
                                                         uint64_t to_insert,
                                                         int pos,
                                                         int len) {
    uint64_t ret;
    asm("bfi.b64 %0, %1, %2, %3, %4;"
        : "=l"(ret)
        : "l"(to_insert), "l"(val), "r"(pos), "r"(len));
    return ret;
  }
};

template <typename T>
struct RadixTypeConfig {};

template <>
struct RadixTypeConfig<float> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType Convert(float v) {
    RadixType x = __float_as_int(v);
    RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;

    return (v == v) ? (x ^ mask) : 0xffffffff;
  }

  static inline __device__ float Deconvert(RadixType v) {
    RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;

    return __int_as_float(v ^ mask);
  }
};

template <>
struct RadixTypeConfig<double> {
  typedef uint64_t RadixType;

  static inline __device__ RadixType Convert(double v) {
    RadixType x = __double_as_longlong(v);
    RadixType mask = -((x >> 63)) | 0x8000000000000000;
    return (v == v) ? (x ^ mask) : 0xffffffffffffffff;
  }

  static inline __device__ double Deconvert(RadixType v) {
    RadixType mask = ((v >> 63) - 1) | 0x8000000000000000;
    return __longlong_as_double(v ^ mask);
  }
};

template <>
struct RadixTypeConfig<int32_t> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType Convert(int32_t v) {
    static_assert(sizeof(int) == 4, "");
    return 2147483648u + v;
  }

  static inline __device__ int32_t Deconvert(RadixType v) {
    return v - 2147483648u;
  }
};

template <>
struct RadixTypeConfig<int64_t> {
  typedef uint64_t RadixType;

  static inline __device__ RadixType Convert(int64_t v) {
    static_assert(sizeof(int64_t) == 8, "");
    return 9223372036854775808ull + v;
  }

  static inline __device__ int64_t Deconvert(RadixType v) {
    return v - 9223372036854775808ull;
  }
};

template <>
struct RadixTypeConfig<phi::dtype::float16> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType Convert(phi::dtype::float16 v) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
    half v_h = v.to_half();
    RadixType x = __half_as_ushort(v_h);
    RadixType mask = (x & 0x00008000) ? 0x0000ffff : 0x00008000;
    return (v_h == v_h) ? (x ^ mask) : 0xffff;
#else
    assert(false);
    return 0u;
#endif
  }

  static inline __device__ phi::dtype::float16 Deconvert(RadixType v) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
    RadixType mask = (v & 0x00008000) ? 0x00008000 : 0x0000ffff;
    return static_cast<phi::dtype::float16>(__ushort_as_half(v ^ mask));
#else
    assert(false);
    return static_cast<phi::dtype::float16>(0);
#endif
  }
};

template <>
struct RadixTypeConfig<phi::dtype::bfloat16> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType Convert(phi::dtype::bfloat16 v) {
    RadixType x = v.x;
    RadixType mask = (x & 0x00008000) ? 0x0000ffff : 0x00008000;
    return (v == v) ? (x ^ mask) : 0xffff;
  }

  static inline __device__ phi::dtype::bfloat16 Deconvert(RadixType v) {
    RadixType mask = (v & 0x00008000) ? 0x00008000 : 0x0000ffff;
    phi::dtype::bfloat16 r;
    r.x = (v ^ mask);
    return r;
  }
};

/*---------------------------Helper Functions------------------*/
__device__ __forceinline__ int GetLaneId() {
  int lane_id;
  asm("mov.s32 %0, %%laneid;" : "=r"(lane_id));
  return lane_id;
}

__device__ __forceinline__ unsigned GetLaneMaskLe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
}

template <typename T, bool KillDependency, class Function>
__device__ void InclusiveBinaryPrefixScan(T* shared_mem,
                                          bool in,
                                          T* out,
                                          Function func) {
  T vote = __ballot_sync(__activemask(), in);
  T index = __popc(GetLaneMaskLe() & vote);
  T carry = __popc(vote);

  int warp = threadIdx.x / 32;

  if (GetLaneId() == 0) {
    shared_mem[warp] = carry;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    int current = 0;
    for (int i = 0; i < blockDim.x / 32; ++i) {
      T v = shared_mem[i];
      shared_mem[i] = func(shared_mem[i], current);
      current = func(current, v);
    }
  }

  __syncthreads();

  if (warp >= 1) {
    index = func(index, shared_mem[warp - 1]);
  }

  *out = index;

  if (KillDependency) {
    __syncthreads();
  }
}

template <typename T, bool KillDependency, class Function>
__device__ void ExclusiveBinaryPrefixScan(
    T* shared_mem, bool in, T* out, T* carry, Function func) {
  InclusiveBinaryPrefixScan<T, false, Function>(shared_mem, in, out, func);

  *out -= (T)in;

  *carry = shared_mem[(blockDim.x + 31) / 32 - 1];

  if (KillDependency) {
    __syncthreads();
  }
}

template <typename T, typename RadixType>
__device__ T FindPattern(const T* input,
                         T* shared_mem,
                         int slice_size,
                         RadixType desired,
                         RadixType desired_mask) {
  if (threadIdx.x < 2) {
    shared_mem[threadIdx.x] = static_cast<T>(0);
  }
  __syncthreads();

  int block_dim = static_cast<int>(blockDim.x);
  int loop = ((slice_size + block_dim - 1) / block_dim * block_dim);
  for (int i = threadIdx.x; i < loop; i += blockDim.x) {
    bool valid = (i < slice_size);
    T v = valid ? input[i] : static_cast<T>(0);

    if (valid && ((RadixTypeConfig<T>::Convert(v) & desired_mask) == desired)) {
      shared_mem[0] = static_cast<T>(1);
      shared_mem[1] = v;
    }

    __syncthreads();

    T found = shared_mem[0];
    T val = shared_mem[1];

    __syncthreads();

    if (found != static_cast<T>(0)) {
      return val;
    }
  }

  assert(false);
  return static_cast<T>(0);
}

template <typename T, typename RadixType, int RadixSize, int RadixBits>
__device__ void RadixCountUsingMask(const T* input,
                                    int counts[RadixSize],
                                    int* shared_mem,
                                    RadixType desired,
                                    RadixType desired_mask,
                                    int radix_digit_pos,
                                    int slice_size) {
#pragma unroll
  for (int i = 0; i < RadixSize; ++i) {
    counts[i] = 0;
  }

  if (threadIdx.x < RadixSize) {
    shared_mem[threadIdx.x] = 0;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < slice_size; i += blockDim.x) {
    RadixType val = RadixTypeConfig<T>::Convert(input[i]);

    bool has_val = ((val & desired_mask) == desired);
    RadixType digit_in_radix =
        Bitfield<RadixType>::GetBitfield(val, radix_digit_pos, RadixBits);

#pragma unroll
    for (uint32_t j = 0; j < RadixSize; ++j) {
      bool vote = has_val && (digit_in_radix == j);
      counts[j] += __popc(__ballot_sync(__activemask(), vote));
    }
  }

  if (GetLaneId() == 0) {
#pragma unroll
    for (uint32_t i = 0; i < RadixSize; ++i) {
      phi::CudaAtomicAdd(&shared_mem[i], counts[i]);
    }
  }

  __syncthreads();

#pragma unroll
  for (uint32_t i = 0; i < RadixSize; ++i) {
    counts[i] = shared_mem[i];
  }

  __syncthreads();
}

template <typename T, typename RadixType, bool Largest>
__device__ void RadixSearch(
    const T* input, int k, int slice_size, int* shared_mem, T* kth_value) {
  int counts[RADIX_SIZE];

  RadixType desired = 0;
  RadixType desired_mask = 0;

  int k_left = k;

#pragma unroll
  for (int digit_pos = sizeof(T) * 8 - RADIX_BITS; digit_pos >= 0;
       digit_pos -= RADIX_BITS) {
    RadixCountUsingMask<T, RadixType, RADIX_SIZE, RADIX_BITS>(input,
                                                              counts,
                                                              shared_mem,
                                                              desired,
                                                              desired_mask,
                                                              digit_pos,
                                                              slice_size);

    auto found_unique = [&](int i, int count) -> bool {
      if (count == 1 && k_left == 1) {
        desired =
            Bitfield<RadixType>::SetBitfield(desired, i, digit_pos, RADIX_BITS);
        desired_mask = Bitfield<RadixType>::SetBitfield(
            desired_mask, RADIX_MASK, digit_pos, RADIX_BITS);

        *kth_value = FindPattern<T, RadixType>(input,
                                               reinterpret_cast<T*>(shared_mem),
                                               slice_size,
                                               desired,
                                               desired_mask);
        return true;
      }
      return false;
    };
    auto found_non_unique = [&](int i, int count) -> bool {
      if (count >= k_left) {
        desired =
            Bitfield<RadixType>::SetBitfield(desired, i, digit_pos, RADIX_BITS);
        desired_mask = Bitfield<RadixType>::SetBitfield(
            desired_mask, RADIX_MASK, digit_pos, RADIX_BITS);

        return true;
      }
      k_left -= count;
      return false;
    };

    if (Largest) {
// Descending order
#pragma unroll
      for (int i = RADIX_SIZE - 1; i >= 0; --i) {
        int count = counts[i];
        if (found_unique(i, count)) {
          return;
        }
        if (found_non_unique(i, count)) {
          break;
        }
      }
    } else {
// Ascending order
#pragma unroll
      for (int i = 0; i < RADIX_SIZE; ++i) {
        int count = counts[i];
        if (found_unique(i, count)) {
          return;
        }
        if (found_non_unique(i, count)) {
          break;
        }
      }
    }
  }

  *kth_value = RadixTypeConfig<T>::Deconvert(desired);
}

template <typename T>
__global__ void GatherKthValue(const T* input,
                               const int k,
                               const int64_t num_rows,
                               const int64_t num_cols,
                               T* output,
                               int64_t* indices) {
  __shared__ int shared_mem[32];
  int row =
      blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
  const T* cur_input = input + row * num_cols;

  // 1. Find the k-th value
  T kth_value = static_cast<T>(0);
  RadixSearch<T, RadixTypeConfig<T>::RadixType, false>(
      cur_input, k, num_cols, shared_mem, &kth_value);
  const auto converted_kth_value = RadixTypeConfig<T>::Convert(kth_value);

  // 2. find the k-th index
  int64_t kth_index = 0;
  bool foundKValue = false;
  for (int64_t i = threadIdx.x; i < num_cols; i += blockDim.x) {
    bool inRange = (i < num_cols);
    T v = inRange ? cur_input[i] : static_cast<T>(0);
    bool isKValue =
        inRange && ((v == kth_value) || (isnan(static_cast<float>(v)) &&
                                         isnan(static_cast<float>(kth_value))));
    if (isKValue) {
      kth_index = i;
      foundKValue = true;
      break;
    }
  }

  if (foundKValue) {
    output[row] = kth_value;
    indices[row] = kth_index;
  }
}

template <typename T>
void LaunchGatherKthValue(const phi::GPUContext& dev_ctx,
                          const T* input_data,
                          const int64_t num_cols,
                          const int64_t num_rows,
                          const int k,
                          T* out_data,
                          int64_t* indices_data) {
  int num_threads = std::min(
      static_cast<int>(round_up(static_cast<int>(num_cols), WARP_SIZE)),
      MAX_NUM_THREADS);
  GatherKthValue<T><<<num_rows, num_threads, 0, dev_ctx.stream()>>>(
      input_data, k, num_rows, num_cols, out_data, indices_data);
}

template <typename T, bool Largest>
__global__ void RadixTopK(const T* input,
                          int k,
                          int slice_num,
                          int slice_size,
                          T* output,
                          int64_t* indices) {
  __shared__ int shared_mem[32];

  // 1. Find the k-th value
  T kth_value = static_cast<T>(0);
  RadixSearch<T, typename RadixTypeConfig<T>::RadixType, Largest>(
      input, k, slice_size, shared_mem, &kth_value);
  const auto converted_kth_value = RadixTypeConfig<T>::Convert(kth_value);

  // 2. Select the value strictly less/greater than kth_value and their indices
  int block_dim = static_cast<int>(blockDim.x);
  int loop = ((slice_size + block_dim - 1) / block_dim * block_dim);
  int write_start = 0;

  for (int i = threadIdx.x; i < loop; i += blockDim.x) {
    bool valid = i < slice_size;
    T v = valid ? input[i] : static_cast<T>(0);
    const auto convertd_v = RadixTypeConfig<T>::Convert(v);
    bool is_top_k;
    if (Largest) {
      is_top_k = valid && (convertd_v > converted_kth_value);
    } else {
      is_top_k = valid && (convertd_v < converted_kth_value);
    }

    int index;
    int carry;
    ExclusiveBinaryPrefixScan<int, true, kps::AddFunctor<int>>(
        shared_mem, is_top_k, &index, &carry, kps::AddFunctor<int>());
    if (is_top_k) {
      int write_index = write_start + index;
      output[write_index] = v;
      indices[write_index] = i;
    }
    write_start += carry;
  }

  // 3. Fill the rest with value == kth_value
  assert(k >= write_start);
  int remain = k - write_start;
  for (int i = threadIdx.x; i < loop; i += blockDim.x) {
    bool valid = i < slice_size;
    T v = valid ? input[i] : static_cast<T>(0);
    const auto convertd_v = RadixTypeConfig<T>::Convert(v);
    bool is_top_k = valid && (convertd_v == converted_kth_value);

    int index;
    int carry;
    ExclusiveBinaryPrefixScan<int, true, kps::AddFunctor<int>>(
        shared_mem, is_top_k, &index, &carry, kps::AddFunctor<int>());
    if (is_top_k && index < remain) {
      int write_index = write_start + index;
      assert(write_index < k);
      output[write_index] = v;
      indices[write_index] = i;
    }

    if (carry >= remain) {
      break;
    }

    remain -= carry;
    write_start += carry;
  }
}
#endif
/*---------------------------Radix TopK End------------------*/

template <typename T, int MaxLength, int BlockSize>
__global__ void AssignGrad(T* x_grad,
                           const int64_t* indices,
                           const T* out_grad,
                           size_t rows,
                           size_t cols,
                           size_t k) {
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      x_grad[i * cols + j] = 0;
    }
    __syncthreads();
    for (size_t j = 0; j < k; ++j) {
      size_t idx = indices[i * k + j];
      x_grad[i * cols + idx] = out_grad[i * k + j];
    }
  }
}

// the grad assign with the axis
template <typename T>
__global__ void AssignGradWithAxis(const T* grad_out,
                                   const int64_t* indices,
                                   T* grad_in,
                                   int pre,
                                   int post,
                                   int raw_height,
                                   int k) {
  // raw_height is the length of topk axis
  for (int i = blockIdx.x; i < pre; i += gridDim.x) {
    int base_index = i * post * k;
    int base_grad = i * post * raw_height;
    for (int j = threadIdx.x; j < raw_height * post; j += blockDim.x) {
      grad_in[base_grad + j] = static_cast<T>(0);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < k * post; j += blockDim.x) {
      int64_t idx_ij = indices[base_index + j];
      int64_t in_ij = base_grad + (idx_ij * post) + (j % post);
      grad_in[in_ij] = grad_out[base_index + j];
    }
  }
}
// use the radix sort for the topk
template <typename T>
bool SortTopk(const phi::GPUContext& ctx,
              const phi::DenseTensor* input_tensor,
              const int64_t num_cols,
              const int64_t num_rows,
              const int k,
              phi::DenseTensor* out_tensor,
              phi::DenseTensor* indices_tensor,
              bool largest = true) {
  auto cu_stream = ctx.stream();

  Tensor input_indices;
  const std::vector<int64_t> dims = {num_rows, num_cols};
  auto dim = phi::make_ddim(dims);
  input_indices.Resize(dim);
  ctx.template Alloc<int64_t>(&input_indices);
  size_t temp_storage_bytes = -1;

  auto ComputeBlockSize = [](int col) {
    if (col > 512)
      return 1024;
    else if (col > 256 && col <= 512)
      return 512;
    else if (col > 128 && col <= 256)
      return 256;
    else if (col > 64 && col <= 128)
      return 128;
    else
      return 64;
  };
  int block_size = ComputeBlockSize(num_cols);

  unsigned int maxGridDimX = ctx.GetCUDAMaxGridDimSize()[0];
  // actually, int num_rows < max_grid_size
  unsigned int grid_size = num_rows < maxGridDimX
                               ? static_cast<unsigned int>(num_rows)
                               : maxGridDimX;
  // Init a index array
  InitIndex<int64_t><<<grid_size, block_size, 0, cu_stream>>>(
      input_indices.data<int64_t>(), num_rows, num_cols);

  // create iter for counting input
  cub::CountingInputIterator<int64_t> counting_iter(0);
  // segment_offset is used for move to next row
  cub::TransformInputIterator<int64_t,
                              SegmentOffsetIter,
                              cub::CountingInputIterator<int64_t>>
      segment_offsets_t(counting_iter, SegmentOffsetIter(num_cols));

  T* sorted_values_ptr;
  int64_t* sorted_indices_ptr;

  Tensor temp_values;
  Tensor temp_indices;

  const T* input = input_tensor->data<T>();
  T* values = out_tensor->data<T>();
  int64_t* indices = ctx.template Alloc<int64_t>(indices_tensor);

  if (k == num_cols) {
    // Doing a full sort.
    sorted_values_ptr = values;
    sorted_indices_ptr = indices;
  } else {
    temp_values.Resize(dim);
    temp_indices.Resize(dim);
    sorted_values_ptr = ctx.template Alloc<T>(&temp_values);
    sorted_indices_ptr = ctx.template Alloc<int64_t>(&temp_indices);
  }

  // Get temp storage buffer size, maybe can allocate a fixed buffer to save
  // time.
  if (largest) {
    auto err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr,
        temp_storage_bytes,
        input,
        sorted_values_ptr,
        input_indices.data<int64_t>(),
        sorted_indices_ptr,
        num_cols * num_rows,
        num_rows,
        segment_offsets_t,
        segment_offsets_t + 1,
        0,
        sizeof(T) * 8,
        cu_stream);
#ifdef __HIPCC__
    if (err != hipSuccess) {
      LOG(ERROR) << "TopKOP failed as could not launch "
                    "hipcub::DeviceSegmentedRadixSort::SortPairsDescending to "
                    "calculate "
                    "temp_storage_bytes, status: "
                 << hipGetErrorString(err);
      return false;
    }
#else
    if (err != cudaSuccess) {
      LOG(ERROR)
          << "TopKOP failed as could not launch "
             "cub::DeviceSegmentedRadixSort::SortPairsDescending to calculate "
             "temp_storage_bytes, status: "
          << cudaGetErrorString(err);
      return false;
    }
#endif
  } else {
    auto err =
        cub::DeviceSegmentedRadixSort::SortPairs(nullptr,
                                                 temp_storage_bytes,
                                                 input,
                                                 sorted_values_ptr,
                                                 input_indices.data<int64_t>(),
                                                 sorted_indices_ptr,
                                                 num_cols * num_rows,
                                                 num_rows,
                                                 segment_offsets_t,
                                                 segment_offsets_t + 1,
                                                 0,
                                                 sizeof(T) * 8,
                                                 cu_stream);
#ifdef __HIPCC__
    if (err != hipSuccess) {
      LOG(ERROR) << "TopKOP failed as could not launch "
                    "hipcub::DeviceSegmentedRadixSort::SortPairs to calculate "
                    "temp_storage_bytes, status: "
                 << hipGetErrorString(err);
      return false;
    }
#else
    if (err != cudaSuccess) {
      LOG(ERROR) << "TopKOP failed as could not launch "
                    "cub::DeviceSegmentedRadixSort::SortPairs to calculate "
                    "temp_storage_bytes, status: "
                 << cudaGetErrorString(err);
      return false;
    }
#endif
  }
  Tensor temp_storage;
  ctx.template Alloc<uint8_t>(&temp_storage, temp_storage_bytes);

  if (largest) {
    auto err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.data<uint8_t>(),
        temp_storage_bytes,
        input,
        sorted_values_ptr,
        input_indices.data<int64_t>(),
        sorted_indices_ptr,
        num_cols * num_rows,
        num_rows,
        segment_offsets_t,
        segment_offsets_t + 1,
        0,
        sizeof(T) * 8,
        cu_stream);
#ifdef __HIPCC__
    if (err != hipSuccess) {
      LOG(ERROR) << "TopKOP failed as could not launch "
                    "hipcub::DeviceSegmentedRadixSort::SortPairsDescending to "
                    "sort input, "
                    "temp_storage_bytes: "
                 << temp_storage_bytes
                 << ", status: " << hipGetErrorString(err);
      return false;
    }
#else
    if (err != cudaSuccess) {
      LOG(ERROR) << "TopKOP failed as could not launch "
                    "cub::DeviceSegmentedRadixSort::SortPairsDescending to "
                    "sort input, "
                    "temp_storage_bytes: "
                 << temp_storage_bytes
                 << ", status: " << cudaGetErrorString(err);
      return false;
    }
#endif
  } else {
    auto err =
        cub::DeviceSegmentedRadixSort::SortPairs(temp_storage.data<uint8_t>(),
                                                 temp_storage_bytes,
                                                 input,
                                                 sorted_values_ptr,
                                                 input_indices.data<int64_t>(),
                                                 sorted_indices_ptr,
                                                 num_cols * num_rows,
                                                 num_rows,
                                                 segment_offsets_t,
                                                 segment_offsets_t + 1,
                                                 0,
                                                 sizeof(T) * 8,
                                                 cu_stream);
#ifdef __HIPCC__
    if (err != hipSuccess) {
      LOG(ERROR) << "TopKOP failed as could not launch "
                    "hipcub::DeviceSegmentedRadixSort::SortPairs to "
                    "sort input, "
                    "temp_storage_bytes: "
                 << temp_storage_bytes
                 << ", status: " << hipGetErrorString(err);
      return false;
    }
#else
    if (err != cudaSuccess) {
      LOG(ERROR) << "TopKOP failed as could not launch "
                    "cub::DeviceSegmentedRadixSort::SortPairs to "
                    "sort input, "
                    "temp_storage_bytes: "
                 << temp_storage_bytes
                 << ", status: " << cudaGetErrorString(err);
      return false;
    }
#endif
  }
  auto& dev = *ctx.eigen_device();
  if (k < num_cols) {
    // copy sliced data to output.
    const Eigen::DSizes<Eigen::DenseIndex, 2> slice_indices{0, 0};
    const Eigen::DSizes<Eigen::DenseIndex, 2> slice_sizes{num_rows, k};
    auto e_indices = phi::EigenMatrix<int64_t>::From(*indices_tensor, dim);
    auto e_tmp_indices = phi::EigenMatrix<int64_t>::From(
        static_cast<const Tensor>(temp_indices));

    std::vector<int> odims = {static_cast<int>(num_rows), static_cast<int>(k)};
    auto dim = phi::make_ddim(odims);
    auto e_values = phi::EigenMatrix<T>::From(*out_tensor, dim);
    auto e_tmp_values =
        phi::EigenMatrix<T>::From(static_cast<const Tensor>(temp_values));

    phi::funcs::EigenSlice<std::decay_t<decltype(dev)>, int64_t, 2>::Eval(
        dev, e_indices, e_tmp_indices, slice_indices, slice_sizes);
    phi::funcs::EigenSlice<std::decay_t<decltype(dev)>, T, 2>::Eval(
        dev, e_values, e_tmp_values, slice_indices, slice_sizes);
  }
  return true;
}
}  // namespace funcs
}  // namespace phi
