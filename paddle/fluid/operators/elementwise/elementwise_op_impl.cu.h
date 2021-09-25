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
#include "paddle/fluid/platform/aligned_vector.h"
#include "paddle/fluid/platform/function_traits.h"

#ifdef __HIPCC__
#define ELEMENTWISE_BLOCK_SIZE 256
#else
#define ELEMENTWISE_BLOCK_SIZE 512
#endif

namespace paddle {
namespace operators {

namespace kps = paddle::operators::kernel_primitives;

enum ElementwiseType { kUnary = 1, kBinary = 2, kTernary = 3, kAny = -1 };

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

template <typename InT, typename OutT>
int GetVectorizedSizeForTensors(
    const std::vector<const framework::Tensor *> &ins,
    const std::vector<framework::Tensor *> &outs) {
  int vec_size = 4;
  for (auto iter = ins.begin(); iter != ins.end(); ++iter) {
    vec_size = std::min<int>(vec_size,
                             platform::GetVectorizedSize((*iter)->data<InT>()));
  }
  for (auto iter = outs.begin(); iter != outs.end(); ++iter) {
    vec_size = std::min<int>(
        vec_size, platform::GetVectorizedSize((*iter)->data<OutT>()));
  }
  return vec_size;
}

template <typename InT, typename OutT, int VecSize, typename Functor, int Arity,
          bool CallElementwiseAny = false>
struct ElementwisePrimitiveCaller {
  __device__ inline void operator()(Functor func, InT (*args)[VecSize],
                                    OutT *result);
};

template <typename InT, typename OutT, int VecSize, typename Functor, int Arity>
struct ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, Arity, true> {
  __device__ inline void operator()(Functor func, InT (*args)[VecSize],
                                    OutT *result) {
    kps::ElementwiseAny<InT, OutT, VecSize, 1, 1, Arity, Functor>(result, args,
                                                                  func);
  }
};

template <typename InT, typename OutT, int VecSize, typename Functor>
struct ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, 1, false> {
  __device__ inline void operator()(Functor func, InT (*args)[VecSize],
                                    OutT *result) {
    kps::ElementwiseUnary<InT, OutT, VecSize, 1, 1, Functor>(result, args[0],
                                                             func);
  }
};

template <typename InT, typename OutT, int VecSize, typename Functor>
struct ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, 2, false> {
  __device__ inline void operator()(Functor func, InT (*args)[VecSize],
                                    OutT *result) {
    kps::ElementwiseBinary<InT, OutT, VecSize, 1, 1, Functor>(result, args[0],
                                                              args[1], func);
  }
};

template <typename InT, typename OutT, int VecSize, typename Functor>
struct ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, 3, false> {
  __device__ inline void operator()(Functor func, InT (*args)[VecSize],
                                    OutT *result) {
    kps::ElementwiseTernary<InT, OutT, VecSize, 1, 1, Functor>(
        result, args[0], args[1], args[2], func);
  }
};

template <typename InT, typename OutT, typename Functor, int Arity, int VecSize,
          bool IsBoundary>
__device__ void DealSegment(
    const framework::Array<const InT *__restrict__, Arity> &in, OutT *out,
    int num, Functor func) {
  InT args[Arity][VecSize];
  OutT result[VecSize];

  int data_offset = VecSize * blockIdx.x * blockDim.x;

#pragma unroll
  for (int i = 0; i < Arity; i++) {
    kps::Init<InT, VecSize>(args[i], static_cast<InT>(1.0f));
    kps::ReadData<InT, VecSize, 1, 1, IsBoundary>(args[i], in[i] + data_offset,
                                                  num);
  }

  const bool kCallElementwiseAny =
      platform::FunctionTraits<Functor>::has_pointer_args;
  ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, Arity,
                             kCallElementwiseAny>()(func, args, result);
  kps::WriteData<OutT, VecSize, 1, 1, IsBoundary>(out + data_offset, result,
                                                  num);
}

template <typename InT, typename OutT, typename Functor, int Arity, int VecSize>
__global__ void ElementVectorizeKernel(
    framework::Array<const InT *__restrict__, Arity> ins, OutT *out, int size,
    Functor func) {
  int data_offset = VecSize * blockIdx.x * blockDim.x;
  int num = size - data_offset;
  // the num this time have to deal with
  if (VecSize * blockDim.x > num) {  // reminder segment
    DealSegment<InT, OutT, Functor, Arity, VecSize, true>(ins, out, num, func);
  } else {  // complete segment
    DealSegment<InT, OutT, Functor, Arity, VecSize, false>(ins, out, num, func);
  }
}

template <typename InT, typename OutT, typename Functor, int Arity, int VecSize>
void ElementwiseCudaKernel(const platform::CUDADeviceContext &ctx,
                           const std::vector<const framework::Tensor *> &ins,
                           std::vector<framework::Tensor *> *outs,
                           Functor func) {
  auto numel = ins[0]->numel();
  int block_size = GetThreadsConfig(ctx, numel, VecSize);
  int grid_size =
      ((numel + VecSize - 1) / VecSize + block_size - 1) / block_size;

  auto stream = ctx.stream();
  OutT *out_data = (*outs)[0]->data<OutT>();
  framework::Array<const InT *__restrict__, Arity> ins_data;
  for (int i = 0; i < Arity; i++) {
    ins_data[i] = ins[i]->data<InT>();
  }
  ElementVectorizeKernel<InT, OutT, Functor, Arity,
                         VecSize><<<grid_size, block_size, 0, stream>>>(
      ins_data, out_data, numel, func);
}

template <ElementwiseType ET, typename InT, typename OutT, typename Functor>
void LaunchSameDimsElementwiseCudaKernel(
    const platform::CUDADeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins,
    std::vector<framework::Tensor *> *outs, Functor func) {
  using Traits = platform::FunctionTraits<Functor>;
  const int kArity =
      Traits::has_pointer_args ? static_cast<int>(ET) : Traits::arity;
  PADDLE_ENFORCE_EQ(ins.size(), kArity,
                    platform::errors::InvalidArgument(
                        "The number of inputs is expected to be equal to the "
                        "arity of functor. But recieved: the number of inputs "
                        "is %d, the arity of functor is %d.",
                        ins.size(), kArity));

  // calculate the max vec_size for all ins and outs
  int vec_size = GetVectorizedSizeForTensors<InT, OutT>(ins, *outs);
  switch (vec_size) {
    case 4:
      ElementwiseCudaKernel<InT, OutT, Functor, kArity, 4>(ctx, ins, outs,
                                                           func);
      break;
    case 2:
      ElementwiseCudaKernel<InT, OutT, Functor, kArity, 2>(ctx, ins, outs,
                                                           func);
      break;
    case 1:
      ElementwiseCudaKernel<InT, OutT, Functor, kArity, 1>(ctx, ins, outs,
                                                           func);
      break;
    default: {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported vectorized size: %d !", vec_size));
      break;
    }
  }
}

}  // namespace operators
}  // namespace paddle
