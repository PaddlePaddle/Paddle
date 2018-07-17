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
// CUDA API uses unsigned long long int, we cannot use uint64_t here.
// It because unsigned long long int is not necessarily uint64_t
USE_CUDA_ATOMIC(Add, unsigned long long int);  // NOLINT

CUDA_ATOMIC_WRAPPER(Add, int64_t) {
  // Here, we check long long int must be int64_t.
  static_assert(sizeof(int64_t) == sizeof(long long int),  // NOLINT
                "long long should be int64");
  return CudaAtomicAdd(
      reinterpret_cast<unsigned long long int*>(address),  // NOLINT
      static_cast<unsigned long long int>(val));           // NOLINT
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
USE_CUDA_ATOMIC(Add, double);
#else
CUDA_ATOMIC_WRAPPER(Add, double) {
  unsigned long long int* address_as_ull =                 // NOLINT
      reinterpret_cast<unsigned long long int*>(address);  // NOLINT
  unsigned long long int old = *address_as_ull, assumed;   // NOLINT

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#ifdef PADDLE_CUDA_FP16
// NOTE(dzhwinter): cuda do not have atomicCAS for half.
// Just use the half address as a unsigned value address and
// do the atomicCAS. According to the value store at high 16 bits
// or low 16 bits, then do a different sum and CAS.
// Given most warp-threads will failed on the atomicCAS, so this
// implemented should be avoided in high concurrency. It's will be
// slower than the way convert value into 32bits and do a full atomicCAS.
CUDA_ATOMIC_WRAPPER(Add, float16) {
  unsigned int* address_as_ui =
      (unsigned int*)(reinterpret_cast<char*>(address) -
                      ((size_t)address & 0x2));
  unsigned int old = *address_as_ui;
  unsigned int sum;
  unsigned int newval;
  unsigned int assumed;

  do {
    assumed = old;
    sum = static_cast<unsigned>(val) + (size_t)address & 0x2
              ? (unsigned)(old >> 16)
              : (unsigned)(old & 0xffff);
    newval = (size_t)address & 0x2 ? (old & 0xffff) | (sum << 16)
                                   : (old & 0xffff0000) | sum;
    old = atomicCAS(address_as_ui, assumed, newval);
  } while (assumed != old);

  auto ret = (size_t)address & 0x2 ? old & 0xffffu : old >> 16;
  return static_cast<float16>(ret);
}

#endif
#endif
}  // namespace platform
}  // namespace paddle
