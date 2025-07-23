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

// The file has been adapted from DeepSeek DeepEP project
// Copyright (c) 2025 DeepSeek
// Licensed under the MIT License -
// https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

#pragma once
#include <cstdint>
#include "paddle/fluid/distributed/collective/deep_ep/kernels/exception.cuh"

#define UNROLLED_WARP_COPY(                                                  \
    UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                   \
  {                                                                          \
    constexpr int kLoopStride = 32 * (UNROLL_FACTOR);                        \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type       \
        unrolled_values[(UNROLL_FACTOR)];                                    \
    auto __src = (SRC);                                                      \
    auto __dst = (DST);                                                      \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride;       \
         __i += kLoopStride) {                                               \
      _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j)      \
          unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32);            \
      _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j)      \
          ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);             \
    }                                                                        \
    for (int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); __i < (N); \
         __i += 32)                                                          \
      ST_FUNC(__dst + __i, LD_FUNC(__src + __i));                            \
  }

namespace deep_ep {

template <int kBytes>
struct VecInt {};
template <>
struct VecInt<1> {
  using vec_t = int8_t;
};
template <>
struct VecInt<2> {
  using vec_t = int16_t;
};
template <>
struct VecInt<4> {
  using vec_t = int;
};
template <>
struct VecInt<8> {
  using vec_t = int64_t;
};
template <>
struct VecInt<16> {
  using vec_t = int4;
};

__device__ __forceinline__ void trap() { asm("trap;"); }

__device__ __forceinline__ void memory_fence() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("fence.acq_rel.sys;" ::: "memory");
#endif
}

__device__ __forceinline__ void memory_fence_gpu() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
#endif
}

__device__ __forceinline__ void memory_fence_cta() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("fence.acq_rel.cta;" ::: "memory");
#endif
}

__device__ __forceinline__ void st_relaxed_sys_global(const int *ptr, int val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("st.relaxed.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val)
               : "memory");
#endif
}

__device__ __forceinline__ void st_release_sys_global(const int *ptr, int val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val)
               : "memory");
#endif
}

__device__ __forceinline__ void st_release_cta(const int *ptr, int val) {
  asm volatile("st.release.cta.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ int ld_acquire_sys_global(const int *ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  int ret;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
#endif
}

__device__ __forceinline__ uint64_t ld_acquire_sys_global(const uint64_t *ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint64_t ret;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
#endif
}

__device__ __forceinline__ int ld_acquire_global(const int *ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  int ret;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
#endif
}

__device__ __forceinline__ int atomic_add_release_sys_global(const int *ptr,
                                                             int value) {
  int ret;
  asm volatile("atom.add.release.sys.global.s32 %0, [%1], %2;"
               : "=r"(ret)
               : "l"(ptr), "r"(value));
  return ret;
}

__device__ __forceinline__ int atomic_add_release_global(const int *ptr,
                                                         int value) {
  int ret;
  asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;"
               : "=r"(ret)
               : "l"(ptr), "r"(value));
  return ret;
}

__device__ __forceinline__ int ld_acquire_cta(const int *ptr) {
  int ret;
  asm volatile("ld.acquire.cta.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ uint8_t ld_na_relaxed(const uint8_t *ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint16_t ret;
  asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b8 %0, [%1];"
               : "=h"(ret)
               : "l"(ptr));
  return static_cast<uint8_t>(ret);
#endif
}

__device__ __forceinline__ uint16_t ld_na_relaxed(const uint16_t *ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint16_t ret;
  asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b16 %0, [%1];"
               : "=h"(ret)
               : "l"(ptr));
  return ret;
#endif
}

__device__ __forceinline__ uint32_t ld_na_relaxed(const uint32_t *ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t ret;
  asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b32 %0, [%1];"
               : "=r"(ret)
               : "l"(ptr));
  return ret;
#endif
}

__device__ __forceinline__ uint64_t ld_na_relaxed(const uint64_t *ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint64_t ret;
  asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];"
               : "=l"(ret)
               : "l"(ptr));
  return ret;
#endif
}

__device__ __forceinline__ int ld_volatile_global(const int *ptr) {
  int ret;
  asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ float ld_volatile_global(const float *ptr) {
  float ret;
  asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(const int64_t *ptr) {
  int64_t ret;
  asm volatile("ld.volatile.global.s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(const uint64_t *ptr) {
  int64_t ret;
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#else
#define DISABLE_AGGRESSIVE_PTX_INSTRS
#endif

#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
#define LD_NC_FUNC "ld.volatile.global"
#endif

// `ld.global.nc.L1::no_allocate` will be translated into
// `LDG.E.NA.[width].CONSTANT` in SASS
template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_nc_global(const dtype_t *ptr) {
  auto ret = ld_nc_global(
      reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t *>(ptr));
  return *reinterpret_cast<dtype_t *>(&ret);
}

template <>
__device__ __forceinline__ uint8_t ld_nc_global(const uint8_t *ptr) {
  uint16_t ret;
  // NOTES: we must use `uint16_t` as inline ASM does not support 8-bit
  // constraint letter (`h` below means unsigned 16-bit)
  asm volatile(LD_NC_FUNC ".u8 %0, [%1];" : "=h"(ret) : "l"(ptr));
  return static_cast<uint8_t>(ret);
}

template <>
__device__ __forceinline__ int ld_nc_global(const int *ptr) {
  int ret;
  asm volatile(LD_NC_FUNC ".s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

template <>
__device__ __forceinline__ int64_t ld_nc_global(const int64_t *ptr) {
  int64_t ret;
  asm volatile(LD_NC_FUNC ".s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

template <>
__device__ __forceinline__ float ld_nc_global(const float *ptr) {
  float ret;
  asm volatile(LD_NC_FUNC ".f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
  return ret;
}

template <>
__device__ __forceinline__ int2 ld_nc_global(const int2 *ptr) {
  int2 ret;
  asm volatile(LD_NC_FUNC ".v2.s32 {%0, %1}, [%2];"
               : "=r"(ret.x), "=r"(ret.y)
               : "l"(ptr));
  return ret;
}

template <>
__device__ __forceinline__ int4 ld_nc_global(const int4 *ptr) {
  int4 ret;
  asm volatile(LD_NC_FUNC ".v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
               : "l"(ptr));
  return ret;
}

__device__ __forceinline__ void st_na_relaxed(const uint8_t *ptr, uint8_t val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("st.relaxed.gpu.global.L1::no_allocate.b8 [%0], %1;"
               :
               : "l"(ptr), "h"(static_cast<uint16_t>(val)));
#endif
}

__device__ __forceinline__ void st_na_relaxed(const uint16_t *ptr,
                                              uint16_t val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("st.relaxed.gpu.global.L1::no_allocate.b16 [%0], %1;"
               :
               : "l"(ptr), "h"(val));
#endif
}

__device__ __forceinline__ void st_na_relaxed(const uint32_t *ptr,
                                              uint32_t val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;"
               :
               : "l"(ptr), "r"(val));
#endif
}

__device__ __forceinline__ void st_na_relaxed(const int *ptr, int val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;"
               :
               : "l"(ptr), "r"(val));
#endif
}

__device__ __forceinline__ void st_na_relaxed(const int4 *ptr, int4 val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile(
      "st.relaxed.gpu.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"
      :
      : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#endif
}

__device__ __forceinline__ void st_na_release(const int *ptr, int val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;"
               :
               : "l"(ptr), "r"(val));
#endif
}

__device__ __forceinline__ void st_na_release(const uint32_t *ptr,
                                              uint32_t val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;"
               :
               : "l"(ptr), "r"(val));
#endif
}

__device__ __forceinline__ void st_na_release(const uint64_t *ptr,
                                              uint64_t val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;"
               :
               : "l"(ptr), "l"(val));
#endif
}

// `st.global.L1::no_allocate` will be translated into `ST.E.NA.[width]` in SASS
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif

template <typename dtype_t>
__device__ __forceinline__ void st_na_global(const dtype_t *ptr,
                                             const dtype_t &value) {
  st_na_global(
      reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t *>(ptr),
      *reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t *>(
          &value));
}

template <>
__device__ __forceinline__ void st_na_global(const int *ptr, const int &value) {
  asm volatile(ST_NA_FUNC ".s32 [%0], %1;" ::"l"(ptr), "r"(value));
}

template <>
__device__ __forceinline__ void st_na_global(const int64_t *ptr,
                                             const int64_t &value) {
  asm volatile(ST_NA_FUNC ".s64 [%0], %1;" ::"l"(ptr), "l"(value));
}

template <>
__device__ __forceinline__ void st_na_global(const float *ptr,
                                             const float &value) {
  asm volatile(ST_NA_FUNC ".f32 [%0], %1;" ::"l"(ptr), "f"(value));
}

template <>
__device__ __forceinline__ void st_na_global(const int4 *ptr,
                                             const int4 &value) {
  asm volatile(ST_NA_FUNC ".v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr),
               "r"(value.x),
               "r"(value.y),
               "r"(value.z),
               "r"(value.w));
}

template <typename dtype_t>
__host__ __device__ dtype_t cell_div(dtype_t a, dtype_t b) {
  return (a + b - 1) / b;
}

template <typename dtype_t>
__host__ __device__ dtype_t align(dtype_t a, dtype_t b) {
  return cell_div<dtype_t>(a, b) * b;
}

__forceinline__ __device__ void get_channel_task_range(int num_tokens,
                                                       int num_sms,
                                                       int sm_id,
                                                       int &token_start_idx,
                                                       int &token_end_idx) {
  int num_tokens_per_sm = cell_div(num_tokens, num_sms);
  token_start_idx = min(num_tokens_per_sm * sm_id, num_tokens);
  token_end_idx = min(token_start_idx + num_tokens_per_sm, num_tokens);
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ dtype_b_t pack2(const dtype_a_t &x,
                                           const dtype_a_t &y) {
  EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t),
                   "Invalid dtypes");
  dtype_b_t packed;
  auto unpacked_ptr = reinterpret_cast<dtype_a_t *>(&packed);
  unpacked_ptr[0] = x, unpacked_ptr[1] = y;
  return packed;
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ void unpack2(const dtype_b_t &packed,
                                        dtype_a_t &x,
                                        dtype_a_t &y) {
  EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t),
                   "Invalid dtypes");
  auto unpacked_ptr = reinterpret_cast<const dtype_a_t *>(&packed);
  x = unpacked_ptr[0], y = unpacked_ptr[1];
}

template <typename dtype_t>
__device__ __forceinline__ dtype_t broadcast(dtype_t &ptr, int src_lane_idx) {
  EP_STATIC_ASSERT(sizeof(dtype_t) % sizeof(int) == 0, "");
  auto send_int_values = reinterpret_cast<int *>(&ptr);
  int recv_int_values[sizeof(dtype_t) / sizeof(int)];
#pragma unroll
  for (int i = 0; i < sizeof(dtype_t) / sizeof(int); ++i)
    recv_int_values[i] =
        __shfl_sync(0xffffffff, send_int_values[i], src_lane_idx);
  return *reinterpret_cast<dtype_t *>(recv_int_values);
}

__forceinline__ __device__ int warp_reduce_sum(int value) {
  value += __shfl_xor_sync(0xffffffff, value, 16);
  value += __shfl_xor_sync(0xffffffff, value, 8);
  value += __shfl_xor_sync(0xffffffff, value, 4);
  value += __shfl_xor_sync(0xffffffff, value, 2);
  value += __shfl_xor_sync(0xffffffff, value, 1);
  return value;
}

__forceinline__ __device__ float half_warp_reduce_max(float value) {
  auto mask = __activemask();
  // The mask be in `{0xffffffff, 0xffff}`
  value = max(value, __shfl_xor_sync(mask, value, 8));
  value = max(value, __shfl_xor_sync(mask, value, 4));
  value = max(value, __shfl_xor_sync(mask, value, 2));
  value = max(value, __shfl_xor_sync(mask, value, 1));
  return value;
}

__forceinline__ __device__ int get_lane_id() {
  int lane_id;
  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
  return lane_id;
}

template <int kNumRanks>
__forceinline__ __device__ void move_fifo_slots(int &head) {
  head = (head + kNumRanks) % NUM_MAX_FIFO_SLOTS;
}

template <int kNumRanks>
__device__ __forceinline__ bool not_finished(int *task, int expected) {
  auto result = false;
  auto lane_id = threadIdx.x % 32;
  if (lane_id < kNumRanks)
    result = ld_volatile_global(task + lane_id) != expected;
  return __any_sync(0xffffffff, result);
}

template <int kNumRanks>
__forceinline__ __device__ void timeout_check(
    int **task_fifo_ptrs, int head, int rank, int expected, int tag = 0) {
  auto start_time = clock64();
  while (not_finished<kNumRanks>(task_fifo_ptrs[rank] + head, expected)) {
    if (clock64() - start_time > NUM_TIMEOUT_CYCLES and threadIdx.x == 0) {
      printf("DeepEP timeout check failed: %d (rank = %d)\n", tag, rank);
      trap();
    }
  }
}

template <int kNumRanks>
__forceinline__ __device__ void barrier_device(int **task_fifo_ptrs,
                                               int head,
                                               int rank,
                                               int tag = 0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  auto thread_id = static_cast<int>(threadIdx.x);
  EP_DEVICE_ASSERT(kNumRanks <= 32);

  if (thread_id < kNumRanks) {
    atomicAdd_system(task_fifo_ptrs[rank] + head + thread_id, FINISHED_SUM_TAG);
    memory_fence();
    atomicSub_system(task_fifo_ptrs[thread_id] + head + rank, FINISHED_SUM_TAG);
  }
  timeout_check<kNumRanks>(task_fifo_ptrs, head, rank, 0, tag);
#endif
}

}  // namespace deep_ep
