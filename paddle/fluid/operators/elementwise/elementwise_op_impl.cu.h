/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/kernel_primitives/kernel_primitives.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/fast_divmod.h"

#ifdef __HIPCC__
#define ELEMENTWISE_BLOCK_SIZE 256
#else
#define ELEMENTWISE_BLOCK_SIZE 512
#endif

namespace paddle {
namespace operators {

enum ElementwiseType { kUnary = 1, kBinary = 2 };

/*
* According to NVIDIA, if number of threads per block is 64/128/256/512,
* cuda performs better. And number of blocks should be greater (at least
* 2x~4x) than number of SMs. Hence, SM count is took into account within
* this function to determine the right number of threads per block.
*/
inline int GetThreadsConfig(const platform::CUDADeviceContext &ctx,
                            int64_t numel, int vec_size) {
  int threads = ELEMENTWISE_BLOCK_SIZE;
  int sm_count = ctx.GetSMCount();
  int active_threads_num = numel / vec_size;
  if (active_threads_num / (sm_count << 1) < ELEMENTWISE_BLOCK_SIZE) {
    // Round up threads number into an exponential multiple of 2, while number
    // of acitve blocks is about twice of SM, to acquire better performance.
    threads = platform::RoundToPowerOfTwo(active_threads_num / (sm_count << 1));
  } else if (active_threads_num / (sm_count << 2) < ELEMENTWISE_BLOCK_SIZE) {
    // Round up threads number into an exponential multiple of 2, while number
    // of acitve blocks is about 4 times of SM, to acquire better performance.
    threads = platform::RoundToPowerOfTwo(active_threads_num / (sm_count << 2));
  }
  // Number of threads per block shall be larger than 64.
  return std::max(64, threads);
}

/*
* Only the address of input data is the multiplier of 1,2,4, vectorized load
* with corresponding multiplier-value is possible. Moreover, the maximum length
* of vectorized load is 128 bits once. Hence, valid length of vectorized load
* shall be determined under both former constraints.
*/
template <typename T>
int GetVectorizedSizeImpl(const T *pointer) {
  constexpr int max_load_bits = 128;
  int valid_vec_size = max_load_bits / CHAR_BIT / sizeof(T);
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec8 =
      std::alignment_of<CudaAlignedVector<T, 8>>::value;  // NOLINT
  constexpr int vec4 =
      std::alignment_of<CudaAlignedVector<T, 4>>::value;  // NOLINT
  constexpr int vec2 =
      std::alignment_of<CudaAlignedVector<T, 2>>::value;  // NOLINT
  if (address % vec8 == 0) {
    /*
    * Currently, decide to deal with no more than 4 data once while adopting
    * vectorization load/store, if performance test shows that dealing with
    * 8 data once in vectorization load/store does get optimized, return code
    * below can be changed into " return std::min(8, valid_vec_size); " .
    */
    return std::min(4, valid_vec_size);
  } else if (address % vec4 == 0) {
    return std::min(4, valid_vec_size);
  } else if (address % vec2 == 0) {
    return std::min(2, valid_vec_size);
  } else {
    return 1;
  }
}

template <typename InT, typename OutT>
int GetVectorizedSize(const std::vector<const framework::Tensor *> &ins,
                      const std::vector<framework::Tensor *> &outs) {
  int vec_size = 4;
  for (auto iter = ins.begin(); iter != ins.end(); ++iter) {
    vec_size =
        std::min<int>(vec_size, GetVectorizedSizeImpl((*iter)->data<InT>()));
  }
  for (auto iter = outs.begin(); iter != outs.end(); ++iter) {
    vec_size =
        std::min<int>(vec_size, GetVectorizedSizeImpl((*iter)->data<OutT>()));
  }
  return vec_size;
}

template <int VecSize, typename InT, typename OutT, typename Functor>
__device__ inline void Binary(const InT *__restrict__ in0,
                              const InT *__restrict__ in1, OutT *out,
                              Functor func, int size) {
  InT args[2][VecSize];
  OutT result[VecSize];
  kernel_primitives::read_data<InT, VecSize, 1, 1>(args[0], in0, size);
  kernel_primitives::read_data<InT, VecSize, 1, 1>(args[1], in1, size);
  kernel_primitives::elementwise_binary<InT, OutT, VecSize, 1, 1, Functor>(
      result, args[0], args[1], func);
  kernel_primitives::write_data<OutT, VecSize, 1, 1>(out, result, size);
}

template <int VecSize, typename InT, typename OutT, typename Functor>
__device__ inline void Unary(const InT *__restrict__ in, OutT *out,
                             Functor func, int size) {
  InT args[VecSize];
  OutT result[VecSize];
  kernel_primitives::read_data<InT, VecSize, 1, 1>(args, in, size);
  kernel_primitives::elementwise_unary<InT, OutT, VecSize, 1, 1, Functor>(
      result, args, func);
  kernel_primitives::write_data<OutT, VecSize, 1, 1>(out, result, size);
}

template <int VecSize, typename InT, typename OutT, typename Functor>
__global__ void ElementVectorizedUnary(const InT *__restrict__ in0, OutT *out,
                                       int size, Functor func) {
  int tid = blockIdx.x * blockDim.x;
  int fix = VecSize * tid;
  int max_size = blockDim.x * VecSize;
  int remain = size - fix;
  int num = remain > max_size ? max_size : remain;
  num = num > 0 ? num : 0;
  Unary<VecSize, InT, OutT, Functor>(in0 + fix, out + fix, func, num);
}

template <int VecSize, typename InT, typename OutT, typename Functor>
__global__ void ElementVectorizedBinary(const InT *__restrict__ in0,
                                        const InT *__restrict__ in1, OutT *out,
                                        int size, Functor func) {
  int tid = blockIdx.x * blockDim.x;
  int fix = VecSize * tid;
  int max_size = blockDim.x * VecSize;
  int remain = size - fix;
  int num = remain > max_size ? max_size : remain;
  num = num > 0 ? num : 0;
  Binary<VecSize, InT, OutT, Functor>(in0 + fix, in1 + fix, out + fix, func,
                                      num);
}

template <typename InT, typename OutT, typename Functor>
void LaunchSameDimsElementwiseUnaryCudaKernel(
    const platform::CUDADeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins,
    std::vector<framework::Tensor *> *outs, Functor func) {
  // calculate the max vec_size for all ins and outs
  auto size = ins[0]->numel();
  const int vec_size = 4;
  int block_size = GetThreadsConfig(ctx, size, vec_size);
  int grid_size =
      ((size + vec_size - 1) / vec_size + block_size - 1) / block_size;
  const InT *in0 = ins[0]->data<InT>();
  OutT *out = (*outs)[0]->data<OutT>();
  // cuda kernel
  auto stream = ctx.stream();

  ElementVectorizedUnary<vec_size><<<grid_size, block_size, 0, stream>>>(
      in0, out, size, func);
}

template <typename InT, typename OutT, typename Functor>
void LaunchSameDimsElementwiseBinaryCudaKernel(
    const platform::CUDADeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins,
    std::vector<framework::Tensor *> *outs, Functor func) {
  // calculate the max vec_size for all ins and outs
  auto size = ins[0]->numel();
  const int vec_size = 4;
  int block_size = GetThreadsConfig(ctx, size, vec_size);
  int grid_size =
      ((size + vec_size - 1) / vec_size + block_size - 1) / block_size;
  const InT *in0 = ins[0]->data<InT>();
  const InT *in1 = ins[1]->data<InT>();
  OutT *out = (*outs)[0]->data<OutT>();
  // cuda kernel
  auto stream = ctx.stream();

  ElementVectorizedBinary<vec_size><<<grid_size, block_size, 0, stream>>>(
      in0, in1, out, size, func);
}

}  // namespace operators
}  // namespace paddle
