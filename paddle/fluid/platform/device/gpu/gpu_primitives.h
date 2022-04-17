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
#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif
#include <stdio.h>
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex.h"
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

#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600)
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

#if CUDA_VERSION >= 10000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
static __device__ __forceinline__ float16 CUDAFP16ToPDFP16(__half x) {
  return *reinterpret_cast<float16 *>(&x);
}

static __device__ __forceinline__ __half PDFP16ToCUDAFP16(float16 x) {
  return *reinterpret_cast<__half *>(&x);
}

CUDA_ATOMIC_WRAPPER(Add, float16) {
  return CUDAFP16ToPDFP16(
      atomicAdd(reinterpret_cast<__half *>(address), PDFP16ToCUDAFP16(val)));
}
#else
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

// The performance of "atomicAdd(half* )" is bad, but for "atomicAdd(half2* )"
// is good. So for fp16 type, we can use "atomicAdd(half2* )" to speed up.
template <typename T, typename std::enable_if<std::is_same<
                          platform::float16, T>::value>::type * = nullptr>
__device__ __forceinline__ void fastAtomicAdd(T *tensor, size_t index,
                                              const size_t numel, T value) {
#if ((CUDA_VERSION < 10000) || \
     (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  CudaAtomicAdd(reinterpret_cast<platform::float16 *>(tensor) + index,
                static_cast<platform::float16>(value));
#else
  // whether the address is 32-byte aligned.
  __half *target_addr = reinterpret_cast<__half *>(tensor + index);
  bool aligned_half2 =
      (reinterpret_cast<std::uintptr_t>(target_addr) % sizeof(__half2) == 0);

  if (aligned_half2 && index < (numel - 1)) {
    __half2 value2;
    value2.x = *reinterpret_cast<__half *>(&value);
    value2.y = __int2half_rz(0);
    atomicAdd(reinterpret_cast<__half2 *>(target_addr), value2);

  } else if (!aligned_half2 && index > 0) {
    __half2 value2;
    value2.x = __int2half_rz(0);
    value2.y = *reinterpret_cast<__half *>(&value);
    atomicAdd(reinterpret_cast<__half2 *>(target_addr - 1), value2);

  } else {
    atomicAdd(reinterpret_cast<__half *>(tensor) + index,
              *reinterpret_cast<__half *>(&value));
  }
#endif
}

template <typename T, typename std::enable_if<!std::is_same<
                          platform::float16, T>::value>::type * = nullptr>
__device__ __forceinline__ void fastAtomicAdd(T *arr, size_t index,
                                              const size_t numel, T value) {
  CudaAtomicAdd(arr + index, value);
}

#ifdef PADDLE_WITH_CUDA
/*
 * One thead block deals with elementwise atomicAdd for vector of len.
 * @in: [x1, x2, x3, ...]
 * @out:[y1+x1, y2+x2, y3+x3, ...]
 * */
template <typename T, typename std::enable_if<!std::is_same<
                          platform::float16, T>::value>::type * = nullptr>
__device__ __forceinline__ void VectorizedAtomicAddPerBlock(
    const int64_t len, int tid, int threads_per_block, const T *in, T *out) {
  for (int i = tid; i < len; i += threads_per_block) {
    CudaAtomicAdd(&out[i], in[i]);
  }
}

// Note: assume that len is even. If len is odd, call fastAtomicAdd directly.
template <typename T, typename std::enable_if<std::is_same<
                          platform::float16, T>::value>::type * = nullptr>
__device__ __forceinline__ void VectorizedAtomicAddPerBlock(
    const int64_t len, int tid, int threads_per_block, const T *in, T *out) {
#if ((CUDA_VERSION < 10000) || \
     (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  for (int i = tid; i < len; i += threads_per_block) {
    CudaAtomicAdd(&out[i], in[i]);
  }
#else
  int i = 0;
  int loops = len / 2 * 2;

  bool aligned_half2 =
      (reinterpret_cast<std::uintptr_t>(out) % sizeof(__half2) == 0);

  if (aligned_half2) {
    for (i = tid * 2; i < loops; i += threads_per_block * 2) {
      __half2 value2;
      T value_1 = in[i];
      T value_2 = in[i + 1];
      value2.x = *reinterpret_cast<__half *>(&value_1);
      value2.y = *reinterpret_cast<__half *>(&value_2);
      atomicAdd(reinterpret_cast<__half2 *>(&out[i]), value2);
    }
    for (; i < len; i += threads_per_block) {
      fastAtomicAdd(out, i, len, in[i]);
    }
  } else {
    for (int i = tid; i < len; i += threads_per_block) {
      fastAtomicAdd(out, i, len, in[i]);
    }
  }
#endif
}
#endif
#endif

// NOTE(zhangbo): cuda do not have atomicCAS for __nv_bfloat16.
inline static __device__ uint32_t bf16_add_to_low_half(uint32_t val, float x) {
  bfloat16 low_half;
  // the bfloat16 in lower 16bits
  low_half.x = static_cast<uint16_t>(val & 0xFFFFu);
  low_half = static_cast<bfloat16>(static_cast<float>(low_half) + x);
  return (val & 0xFFFF0000u) | low_half.x;
}

inline static __device__ uint32_t bf16_add_to_high_half(uint32_t val, float x) {
  bfloat16 high_half;
  // the bfloat16 in higher 16bits
  high_half.x = static_cast<uint16_t>(val >> 16);
  high_half = static_cast<bfloat16>(static_cast<float>(high_half) + x);
  return (val & 0xFFFFu) | (static_cast<uint32_t>(high_half.x) << 16);
}

#if CUDA_VERSION >= 11000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
static __device__ __forceinline__ bfloat16 CUDABF16ToPDBF16(__nv_bfloat16 x) {
  return *reinterpret_cast<bfloat16 *>(&x);
}

static __device__ __forceinline__ __nv_bfloat16 PDBF16ToCUDABF16(bfloat16 x) {
  return *reinterpret_cast<__nv_bfloat16 *>(&x);
}

CUDA_ATOMIC_WRAPPER(Add, bfloat16) {
  return CUDABF16ToPDBF16(atomicAdd(reinterpret_cast<__nv_bfloat16 *>(address),
                                    PDBF16ToCUDABF16(val)));
}
#else
CUDA_ATOMIC_WRAPPER(Add, bfloat16) {
  // concrete packed bfloat16 value may exsits in lower or higher 16bits
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
    // the bfloat16 value stay at lower 16 bits of the address.
    do {
      assumed = old;
      old = atomicCAS(address_as_ui, assumed,
                      bf16_add_to_low_half(assumed, val_f));
    } while (old != assumed);
    bfloat16 ret;
    ret.x = old & 0xFFFFu;
    return ret;
  } else {
    // the bfloat16 value stay at higher 16 bits of the address.
    do {
      assumed = old;
      old = atomicCAS(address_as_ui, assumed,
                      bf16_add_to_high_half(assumed, val_f));
    } while (old != assumed);
    bfloat16 ret;
    ret.x = old >> 16;
    return ret;
  }
}
#endif

CUDA_ATOMIC_WRAPPER(Add, complex<float>) {
  float *real = reinterpret_cast<float *>(address);
  float *imag = real + 1;
  return complex<float>(CudaAtomicAdd(real, val.real),
                        CudaAtomicAdd(imag, val.imag));
}

CUDA_ATOMIC_WRAPPER(Add, complex<double>) {
  double *real = reinterpret_cast<double *>(address);
  double *imag = real + 1;
  return complex<double>(CudaAtomicAdd(real, val.real),
                         CudaAtomicAdd(imag, val.imag));
}

// For atomicMax
USE_CUDA_ATOMIC(Max, int);
USE_CUDA_ATOMIC(Max, unsigned int);
// CUDA API uses unsigned long long int, we cannot use uint64_t here.
// It because unsigned long long int is not necessarily uint64_t
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350)
USE_CUDA_ATOMIC(Max, unsigned long long int);  // NOLINT
#else
CUDA_ATOMIC_WRAPPER(Max, unsigned long long int) {  // NOLINT
  if (*address >= val) {
    return *address;
  }

  unsigned long long int old = *address, assumed;  // NOLINT

  do {
    assumed = old;
    if (assumed >= val) {
      break;
    }

    old = atomicCAS(address, assumed, val);
  } while (assumed != old);
}
#endif

CUDA_ATOMIC_WRAPPER(Max, int64_t) {
  // Here, we check long long int must be int64_t.
  static_assert(sizeof(int64_t) == sizeof(long long int),  // NOLINT
                "long long should be int64");
  long long int res = *address;  // NOLINT
  while (val > res) {
    long long int old = res;                                           // NOLINT
    res = (long long int)atomicCAS((unsigned long long int *)address,  // NOLINT
                                   (unsigned long long int)old,        // NOLINT
                                   (unsigned long long int)val);       // NOLINT
    if (res == old) {
      break;
    }
  }
  return res;
}

CUDA_ATOMIC_WRAPPER(Max, float) {
  if (*address >= val) {
    return *address;
  }

  int *const address_as_i = reinterpret_cast<int *>(address);
  int old = *address_as_i, assumed;

  do {
    assumed = old;
    if (__int_as_float(assumed) >= val) {
      break;
    }

    old = atomicCAS(address_as_i, assumed, __float_as_int(val));
  } while (assumed != old);

  return __int_as_float(old);
}

CUDA_ATOMIC_WRAPPER(Max, double) {
  if (*address >= val) {
    return *address;
  }

  unsigned long long int *const address_as_ull =            // NOLINT
      reinterpret_cast<unsigned long long int *>(address);  // NOLINT
  unsigned long long int old = *address_as_ull, assumed;    // NOLINT

  do {
    assumed = old;
    if (__longlong_as_double(assumed) >= val) {
      break;
    }

    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
  } while (assumed != old);

  return __longlong_as_double(old);
}

// For atomicMin
USE_CUDA_ATOMIC(Min, int);
USE_CUDA_ATOMIC(Min, unsigned int);
// CUDA API uses unsigned long long int, we cannot use uint64_t here.
// It because unsigned long long int is not necessarily uint64_t
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350)
USE_CUDA_ATOMIC(Min, unsigned long long int);  // NOLINT
#else
CUDA_ATOMIC_WRAPPER(Min, unsigned long long int) {  // NOLINT
  if (*address <= val) {
    return *address;
  }

  unsigned long long int old = *address, assumed;  // NOLINT

  do {
    assumed = old;
    if (assumed <= val) {
      break;
    }

    old = atomicCAS(address, assumed, val);
  } while (assumed != old);
}
#endif

CUDA_ATOMIC_WRAPPER(Min, int64_t) {
  // Here, we check long long int must be int64_t.
  static_assert(sizeof(int64_t) == sizeof(long long int),  // NOLINT
                "long long should be int64");
  long long int res = *address;  // NOLINT
  while (val < res) {
    long long int old = res;                                           // NOLINT
    res = (long long int)atomicCAS((unsigned long long int *)address,  // NOLINT
                                   (unsigned long long int)old,        // NOLINT
                                   (unsigned long long int)val);       // NOLINT
    if (res == old) {
      break;
    }
  }
  return res;
}

CUDA_ATOMIC_WRAPPER(Min, float) {
  if (*address <= val) {
    return *address;
  }

  int *const address_as_i = reinterpret_cast<int *>(address);
  int old = *address_as_i, assumed;

  do {
    assumed = old;
    if (__int_as_float(assumed) <= val) {
      break;
    }

    old = atomicCAS(address_as_i, assumed, __float_as_int(val));
  } while (assumed != old);

  return __int_as_float(old);
}

CUDA_ATOMIC_WRAPPER(Min, double) {
  if (*address <= val) {
    return *address;
  }

  unsigned long long int *const address_as_ull =            // NOLINT
      reinterpret_cast<unsigned long long int *>(address);  // NOLINT
  unsigned long long int old = *address_as_ull, assumed;    // NOLINT

  do {
    assumed = old;
    if (__longlong_as_double(assumed) <= val) {
      break;
    }

    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
  } while (assumed != old);

  return __longlong_as_double(old);
}

}  // namespace platform
}  // namespace paddle
