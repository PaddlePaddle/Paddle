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
#include <stdio.h>
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace platform {

#define CUDA_ATOMIC_WRAPPER(op, T) \
  __device__ __forceinline__ T CudaAtomic##op(T *address, const T val)

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
      reinterpret_cast<unsigned long long int *>(address),  // NOLINT
      static_cast<unsigned long long int>(val));            // NOLINT
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
USE_CUDA_ATOMIC(Add, double);
#else
CUDA_ATOMIC_WRAPPER(Add, double) {
  unsigned long long int *address_as_ull =                  // NOLINT
      reinterpret_cast<unsigned long long int *>(address);  // NOLINT
  unsigned long long int old = *address_as_ull, assumed;    // NOLINT

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

#ifdef PADDLE_CUDA_FP16
// NOTE(dzhwinter): cuda do not have atomicCAS for half.
// Just use the half address as a unsigned value address and
// do the atomicCAS. According to the value store at high 16 bits
// or low 16 bits, then do a different sum and CAS.
// Given most warp-threads will failed on the atomicCAS, so this
// implemented should be avoided in high concurrency. It's will be
// slower than the way convert value into 32bits and do a full atomicCAS.

// convert the value into float and do the add arithmetic.
// then store the result into a uint32.
inline static __device__ uint32_t add_to_low_half(uint32_t val, float x) {
  float16 low_half;
  // the float16 in lower 16bits
  low_half.x = static_cast<uint16_t>(val & 0xFFFFu);
  low_half = static_cast<float16>(static_cast<float>(low_half) + x);
  return (val & 0xFFFF0000u) | low_half.x;
}

inline static __device__ uint32_t add_to_high_half(uint32_t val, float x) {
  float16 high_half;
  // the float16 in higher 16bits
  high_half.x = static_cast<uint16_t>(val >> 16);
  high_half = static_cast<float16>(static_cast<float>(high_half) + x);
  return (val & 0xFFFFu) | (static_cast<uint32_t>(high_half.x) << 16);
}

CUDA_ATOMIC_WRAPPER(Add, float16) {
  // concrete packed float16 value may exsits in lower or higher 16bits
  // of the 32bits address.
  uint32_t *address_as_ui = reinterpret_cast<uint32_t *>(
      reinterpret_cast<char *>(address) -
      (reinterpret_cast<uintptr_t>(address) & 0x02));
  float val_f = static_cast<float>(val);
  uint32_t old = *address_as_ui;
  uint32_t sum;
  uint32_t newval;
  uint32_t assumed;
  if (((uintptr_t)address & 0x02) == 0) {
    // the float16 value stay at lower 16 bits of the address.
    do {
      assumed = old;
      old = atomicCAS(address_as_ui, assumed, add_to_low_half(assumed, val_f));
    } while (old != assumed);
    float16 ret;
    ret.x = old & 0xFFFFu;
    return ret;
  } else {
    // the float16 value stay at higher 16 bits of the address.
    do {
      assumed = old;
      old = atomicCAS(address_as_ui, assumed, add_to_high_half(assumed, val_f));
    } while (old != assumed);
    float16 ret;
    ret.x = old >> 16;
    return ret;
  }
}

#endif
}  // namespace platform
}  // namespace paddle
