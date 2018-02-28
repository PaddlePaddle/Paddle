/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <cuda.h>

namespace paddle {
namespace platform {

#define CUDA_ATOMIC_WRAPPER(op, T) \
  __device__ __forceinline__ T CudaAtomic##op(T* address, const T val)

#define USE_CUDA_ATOMIC(op, T) \
  CUDA_ATOMIC_WRAPPER(op, T) { return atomic##op(address, val); }

// Default thread count per block(or block size).
// TODO(typhoonzero): need to benchmark against setting this value
//                    to 1024.
constexpr int PADDLE_CUDA_NUM_THREADS = 512;

// For atomicAdd.
USE_CUDA_ATOMIC(Add, float);
USE_CUDA_ATOMIC(Add, int);
USE_CUDA_ATOMIC(Add, unsigned int);
USE_CUDA_ATOMIC(Add, unsigned long long int);

CUDA_ATOMIC_WRAPPER(Add, int64_t) {
  static_assert(sizeof(int64_t) == sizeof(long long int),
                "long long should be int64");
  return CudaAtomicAdd(reinterpret_cast<unsigned long long int*>(address),
                       static_cast<unsigned long long int>(val));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
USE_CUDA_ATOMIC(Add, double);
#else
CUDA_ATOMIC_WRAPPER(Add, double) {
  unsigned long long int* address_as_ull =
      reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

// __shfl_down has been deprecated as of CUDA 9.0.
#if CUDA_VERSION < 9000
template <typename T>
__forceinline__ __device__ T __shfl_down_sync(unsigned, T val, int delta) {
  return __shfl_down(val, delta);
}
#define CREATE_SHFL_MASK(mask, predicate) mask = 0u;
#else
#define FULL_WARP_MASK 0xFFFFFFFF
#define CREATE_SHFL_MASK(mask, predicate) \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#endif

template <typename T>
__device__ T reduceSum(T val, int tid, int len) {
  // TODO(zcd): The warp size should be taken from the
  // parameters of the GPU but not specified as 32 simply.
  // To make the reduceSum more efficiently,
  // I use Warp-Level Parallelism and assume the Warp size
  // is 32 which may be different for different GPU,
  // but most card's warp size is 32.
  __shared__ T shm[32];
  const int warpSize = 32;
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < len);

  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(mask, val, offset);

  if (tid < warpSize) shm[tid] = 0;

  __syncthreads();

  if (tid % warpSize == 0) {
    shm[tid / warpSize] = val;
  }

  CREATE_SHFL_MASK(mask, tid < warpSize);

  if (tid < warpSize) {
    val = shm[tid];
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      val += __shfl_down_sync(mask, val, offset);
  }

  return val;
}

}  // namespace platform
}  // namespace paddle
