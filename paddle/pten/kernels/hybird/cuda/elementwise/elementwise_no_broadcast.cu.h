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

#include "paddle/pten/kernels/hybird/cuda/elementwise/elementwise_common.cu.h"

#ifdef __HIPCC__
#define ELEMENTWISE_BLOCK_SIZE 256
#else
#define ELEMENTWISE_BLOCK_SIZE 512
#endif

namespace pten {

/*
* According to NVIDIA, if number of threads per block is 64/128/256/512,
* cuda performs better. And number of blocks should be greater (at least
* 2x~4x) than number of SMs. Hence, SM count is took into account within
* this function to determine the right number of threads per block.
*/
inline int GetThreadsConfig(const paddle::platform::CUDADeviceContext &ctx,
                            int64_t numel,
                            int vec_size) {
  int threads = ELEMENTWISE_BLOCK_SIZE;
  int sm_count = ctx.GetSMCount();
  int active_threads_num = numel / vec_size;
  if (active_threads_num / (sm_count << 1) < ELEMENTWISE_BLOCK_SIZE) {
    // Round up threads number into an exponential multiple of 2, while number
    // of acitve blocks is about twice of SM, to acquire better performance.
    threads = paddle::platform::RoundToPowerOfTwo(active_threads_num /
                                                  (sm_count << 1));
  } else if (active_threads_num / (sm_count << 2) < ELEMENTWISE_BLOCK_SIZE) {
    // Round up threads number into an exponential multiple of 2, while number
    // of acitve blocks is about 4 times of SM, to acquire better performance.
    threads = paddle::platform::RoundToPowerOfTwo(active_threads_num /
                                                  (sm_count << 2));
  }
  // Number of threads per block shall be larger than 64.
  return std::max(64, threads);
}

template <typename InT,
          typename OutT,
          typename Functor,
          int Arity,
          int NumOuts,
          int VecSize,
          bool IsBoundary>
__device__ void VectorizedElementwiseKernelImpl(
    const paddle::framework::Array<const InT *__restrict__, Arity> &in,
    paddle::framework::Array<OutT *, NumOuts> outs,
    int num,
    int data_offset,
    Functor func) {
  InT args[Arity][VecSize];
  OutType<OutT, NumOuts> result[VecSize];

#pragma unroll
  for (int i = 0; i < Arity; i++) {
    kps::Init<InT, VecSize>(args[i], static_cast<InT>(1.0f));
    kps::ReadData<InT, VecSize, 1, 1, IsBoundary>(
        args[i], in[i] + data_offset, num);
  }

  constexpr bool kCallElementwiseAny =
      paddle::platform::FunctionTraits<Functor>::has_pointer_args;
  ElementwisePrimitiveCaller<InT,
                             OutType<OutT, NumOuts>,
                             VecSize,
                             Functor,
                             Arity,
                             kCallElementwiseAny>()(func, args, result);

  ElementwiseWriteDataCaller<OutT, VecSize, IsBoundary, NumOuts>()(
      outs, result, data_offset, num);
}

template <typename InT,
          typename OutT,
          typename Functor,
          int Arity,
          int NumOuts,
          int VecSize>
__global__ void VectorizedElementwiseKernel(
    paddle::framework::Array<const InT *__restrict__, Arity> ins,
    paddle::framework::Array<OutT *, NumOuts> outs,
    int size,
    int main_offset,
    Functor func) {
  int data_offset = BLOCK_ID_X * BLOCK_NUM_X * VecSize;
  int stride = BLOCK_NUM_X * GRID_NUM_X * VecSize;
  for (; data_offset < main_offset; data_offset += stride) {
    VectorizedElementwiseKernelImpl<InT,
                                    OutT,
                                    Functor,
                                    Arity,
                                    NumOuts,
                                    VecSize,
                                    false>(
        ins, outs, VecSize * BLOCK_NUM_X, data_offset, func);
  }

  int num = size - data_offset;
  if (num > 0) {
    VectorizedElementwiseKernelImpl<InT,
                                    OutT,
                                    Functor,
                                    Arity,
                                    NumOuts,
                                    VecSize,
                                    true>(ins, outs, num, data_offset, func);
  }
}

template <typename InT, typename OutT>
int GetVectorizedSizeForTensors(const std::vector<const DenseTensor *> &ins,
                                const std::vector<DenseTensor *> &outs) {
  int vec_size = 4;
  for (auto iter = ins.begin(); iter != ins.end(); ++iter) {
    vec_size = std::min<int>(
        vec_size, paddle::platform::GetVectorizedSize((*iter)->data<InT>()));
  }
  for (auto iter = outs.begin(); iter != outs.end(); ++iter) {
    vec_size = std::min<int>(
        vec_size, paddle::platform::GetVectorizedSize((*iter)->data<OutT>()));
  }
  return vec_size;
}

template <typename InT,
          typename OutT,
          typename Functor,
          int Arity,
          int NumOuts,
          int VecSize>
void ElementwiseCudaKernel(const paddle::platform::CUDADeviceContext &ctx,
                           const std::vector<const DenseTensor *> &ins,
                           std::vector<DenseTensor *> *outs,
                           Functor func) {
  auto numel = ins[0]->numel();
  int block_size = GetThreadsConfig(ctx, numel, VecSize);
  int grid_size =
      ((numel + VecSize - 1) / VecSize + block_size - 1) / block_size;
  auto stream = ctx.stream();
  paddle::framework::Array<const InT *__restrict__, Arity> ins_data;
  paddle::framework::Array<OutT *, NumOuts> outs_data;

  for (int i = 0; i < Arity; ++i) {
    ins_data[i] = ins[i]->data<InT>();
  }
  for (int i = 0; i < NumOuts; ++i) {
    outs_data[i] = (*outs)[i]->mutable_data<OutT>();
  }
#ifdef PADDLE_WITH_XPU2
  block_size = 128;
  grid_size = 8;
  int main_offset = (numel / (VecSize * block_size)) * VecSize * block_size;
  VectorizedElementwiseKernel<InT,
                              OutT,
                              Functor,
                              Arity,
                              NumOuts,
                              VecSize><<<grid_size, block_size, 0, stream>>>(
      ins_data, outs_data, numel, main_offset, func);
#else
  int main_offset = (numel / (VecSize * block_size)) * VecSize * block_size;
  VectorizedElementwiseKernel<InT,
                              OutT,
                              Functor,
                              Arity,
                              NumOuts,
                              VecSize><<<grid_size, block_size, 0, stream>>>(
      ins_data, outs_data, numel, main_offset, func);
#endif
}

template <ElementwiseType ET,
          typename InT,
          typename OutT,
          typename Functor,
          int NumOuts = 1>
void LaunchSameDimsElementwiseCudaKernel(
    const paddle::platform::CUDADeviceContext &ctx,
    const std::vector<const DenseTensor *> &ins,
    std::vector<DenseTensor *> *outs,
    Functor func) {
  using Traits = paddle::platform::FunctionTraits<Functor>;
  const int kArity = Traits::has_pointer_args ? static_cast<int>(ET) : Traits::arity;
  PADDLE_ENFORCE_EQ(ins.size(), kArity,
                    paddle::platform::errors::InvalidArgument(
                        "The number of inputs is expected to be equal to the "
                        "arity of functor. But recieved: the number of inputs "
                        "is %d, the arity of functor is %d.",
                        ins.size(),
                        kArity));
  PADDLE_ENFORCE_EQ(outs->size(), NumOuts,
                    paddle::platform::errors::InvalidArgument(
                    "Number of outputs shall equal to number of functions, "
                    "but number of outputs is %d, of functions is %d.",
                    outs->size(),
                    NumOuts));

  if (NumOuts > 1) {
    for (int i = 1; i < NumOuts; ++i) {
      PADDLE_ENFORCE_EQ(
          (*outs)[i]->dims(),
          (*outs)[0]->dims(),
          paddle::platform::errors::InvalidArgument(
              "The shape of each output tensor shall be identical yet, "
              "but %dth output tensor`s shape is not.",
              i));
    }
  }

  // calculate the max vec_size for all ins and outs
  int vec_size = GetVectorizedSizeForTensors<InT, OutT>(ins, *outs);
  switch (vec_size) {
    case 4:
      ElementwiseCudaKernel<InT, OutT, Functor, kArity, NumOuts, 4>(
          ctx, ins, outs, func);
      break;
    case 2:
      ElementwiseCudaKernel<InT, OutT, Functor, kArity, NumOuts, 2>(
          ctx, ins, outs, func);
      break;
    case 1:
      ElementwiseCudaKernel<InT, OutT, Functor, kArity, NumOuts, 1>(
          ctx, ins, outs, func);
      break;
    default: {
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported vectorized size: %d !", vec_size));
      break;
    }
  }
}

}  // namespace pten
