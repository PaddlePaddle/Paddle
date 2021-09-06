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
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/fast_divmod.h"
#include "paddle/fluid/platform/function_traits.h"

#ifdef __HIPCC__
#define ELEMENTWISE_BLOCK_SIZE 256
#else
#define ELEMENTWISE_BLOCK_SIZE 512
#endif

namespace paddle {
namespace operators {

enum ElementwiseType { kUnary = 1, kBinary = 2, kTernary = 3, kUnknown = -1 };

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

template <typename InT, typename OutT, int VecSize, int Arity>
struct ElementwiseArgsWrapper {
  using InVecType = platform::AlignedVector<InT, VecSize>;
  using OutVecType = platform::AlignedVector<OutT, VecSize>;

  const InT *__restrict__ in_data[Arity];
  OutT *out_data;
  uint32_t scalar_cal_offset;

  HOSTDEVICE ElementwiseArgsWrapper(
      const std::vector<const framework::Tensor *> &ins,
      std::vector<framework::Tensor *> *outs, uint32_t scalar_cal_offset)
      : scalar_cal_offset(scalar_cal_offset) {
#pragma unroll
    for (int i = 0; i < Arity; ++i) {
      in_data[i] = ins[i]->data<InT>();
    }
    out_data = (*outs)[0]->data<OutT>();
  }

  inline __device__ void LoadVectorizedData(InVecType vec_args[], int tid) {
#pragma unroll
    for (int i = 0; i < Arity; ++i) {
      const InVecType *in_vec_data =
          reinterpret_cast<const InVecType *>(in_data[i]);
      vec_args[i] = in_vec_data[tid];
    }
  }

  inline __device__ void LoadScalarizedData(InT args[], int tid) {
#pragma unroll
    for (int i = 0; i < Arity; ++i) {
      args[i] = in_data[i][tid + scalar_cal_offset];
    }
  }

  inline __device__ void StoreVectorizedData(OutVecType res, int tid) {
    OutVecType *out_vec = reinterpret_cast<OutVecType *>(out_data);
    out_vec[tid] = res;
  }

  inline __device__ void StoreScalarizedData(OutT res, int tid) {
    out_data[tid + scalar_cal_offset] = res;
  }
};

template <typename InT, typename OutT, int VecSize, int Arity, typename Functor>
__device__ inline void VectorizedKernelImpl(
    ElementwiseArgsWrapper<InT, OutT, VecSize, Arity> wrapper, Functor func,
    int tid) {
  using InVecType = platform::AlignedVector<InT, VecSize>;
  using OutVecType = platform::AlignedVector<OutT, VecSize>;

  InVecType ins_vec[Arity];
  OutVecType out_vec;

  InT *ins_ptr[Arity];
#pragma unroll
  for (int i = 0; i < Arity; ++i) {
    ins_ptr[i] = reinterpret_cast<InT *>(&(ins_vec[i]));
  }
  wrapper.LoadVectorizedData(ins_vec, tid);

#pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    InT ins[Arity];
#pragma unroll
    for (int j = 0; j < Arity; ++j) {
      ins[j] = ins_ptr[j][i];
    }
    out_vec.val[i] = platform::CallFunctor<InT, OutT, Functor>(func, ins);
  }
  // store
  wrapper.StoreVectorizedData(out_vec, tid);
}

template <typename InT, typename OutT, int VecSize, int Arity, typename Functor>
__device__ inline void ScalarKernelImpl(
    ElementwiseArgsWrapper<InT, OutT, VecSize, Arity> wrapper, Functor func,
    int tid) {
  InT ins[Arity];
  wrapper.LoadScalarizedData(ins, tid);
  OutT out = platform::CallFunctor<InT, OutT, Functor>(func, ins);
  wrapper.StoreScalarizedData(out, tid);
}

template <typename InT, typename OutT, int VecSize, int Arity, typename Functor>
__global__ void VectorizedElementwiseKernel(
    ElementwiseArgsWrapper<InT, OutT, VecSize, Arity> wrapper, int main_tid,
    int tail_tid, Functor func) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < main_tid) {
    VectorizedKernelImpl<InT, OutT, VecSize, Arity, Functor>(wrapper, func,
                                                             tid);
  }
  if (tid < tail_tid) {
    ScalarKernelImpl<InT, OutT, VecSize, Arity, Functor>(wrapper, func, tid);
  }
}

template <typename InT, typename OutT, int Arity, typename Functor>
__global__ void ScalarElementwiseKernel(
    ElementwiseArgsWrapper<InT, OutT, 1, Arity> wrapper, int numel,
    Functor func) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numel) {
    ScalarKernelImpl<InT, OutT, 1, Arity, Functor>(wrapper, func, tid);
  }
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
  auto numel = ins[0]->numel();
  int vec_size = GetVectorizedSizeForTensors<InT, OutT>(ins, *outs);
  int block_size = GetThreadsConfig(ctx, numel, vec_size);
  int grid_size =
      ((numel + vec_size - 1) / vec_size + block_size - 1) / block_size;
  int main_tid = numel / vec_size;
  int tail_tid = numel % vec_size;
  uint32_t vec_len = main_tid * vec_size;

  auto stream = ctx.stream();
  switch (vec_size) {
    case 4: {
      auto wrapper =
          ElementwiseArgsWrapper<InT, InT, 4, kArity>(ins, outs, vec_len);
      VectorizedElementwiseKernel<
          InT, InT, 4, kArity, Functor><<<grid_size, block_size, 0, stream>>>(
          wrapper, main_tid, tail_tid, func);
      break;
    }
    case 2: {
      auto wrapper =
          ElementwiseArgsWrapper<InT, InT, 2, kArity>(ins, outs, vec_len);
      VectorizedElementwiseKernel<
          InT, InT, 2, kArity, Functor><<<grid_size, block_size, 0, stream>>>(
          wrapper, main_tid, tail_tid, func);
      break;
    }
    case 1: {
      auto wrapper = ElementwiseArgsWrapper<InT, InT, 1, kArity>(ins, outs, 0);
      ScalarElementwiseKernel<InT, InT, kArity,
                              Functor><<<grid_size, block_size, 0, stream>>>(
          wrapper, numel, func);
      break;
    }
    default: {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported vectorized size: %d !", vec_size));
      break;
    }
  }
}

}  // namespace operators
}  // namespace paddle
