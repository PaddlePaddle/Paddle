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

#include "paddle/fluid/framework/pten_utils.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/kernel_primitives/kernel_primitives.h"
#include "paddle/fluid/platform/aligned_vector.h"
#include "paddle/fluid/platform/function_traits.h"

// only can include the headers in paddle/top/api dirs
#include "paddle/pten/api/lib/utils/tensor_utils.h"
#include "paddle/pten/include/core.h"
#include "paddle/pten/kernels/functions/cuda/elementwise/elementwise.h"

#ifdef __HIPCC__
#define ELEMENTWISE_BLOCK_SIZE 256
#else
#define ELEMENTWISE_BLOCK_SIZE 512
#endif

namespace paddle {
namespace operators {

namespace kps = paddle::operators::kernel_primitives;

<<<<<<< 9c59170353fcd17b44c7de560bd56760cbd5786b
using ElementwiseType = pten::ElementwiseType;
=======
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
__device__ void ElementVectorizeKernelImpl(
    const framework::Array<const InT *__restrict__, Arity> &in, OutT *out,
    int num, int data_offset, Functor func) {
  InT args[Arity][VecSize];
  OutT result[VecSize];

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
    int main_offset, Functor func) {
  int data_offset = BLOCK_ID_X * BLOCK_NUM_X * VecSize;
  int stride = BLOCK_NUM_X * GRID_NUM_X * VecSize;
  for (; data_offset < main_offset; data_offset += stride) {
    ElementVectorizeKernelImpl<InT, OutT, Functor, Arity, VecSize, false>(
        ins, out, VecSize * BLOCK_NUM_X, data_offset, func);
  }

  int num = size - data_offset;
  if (num > 0) {
    ElementVectorizeKernelImpl<InT, OutT, Functor, Arity, VecSize, true>(
        ins, out, num, data_offset, func);
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
#ifdef PADDLE_WITH_XPU2
  block_size = 128;
  grid_size = 8;
  int main_tid = numel / (VecSize * block_size);
  ElementVectorizeKernel<InT, OutT, Functor, Arity,
                         VecSize><<<grid_size, block_size, stream>>>(
      ins_data, out_data, numel, main_tid * block_size * VecSize, func);
#else
  int main_tid = numel / (VecSize * block_size);
  ElementVectorizeKernel<InT, OutT, Functor, Arity,
                         VecSize><<<grid_size, block_size, 0, stream>>>(
      ins_data, out_data, numel, main_tid * block_size * VecSize, func);
#endif
}
>>>>>>> modified the elementwise_op_broadcast and elementwise_op_impl for xpu2

template <ElementwiseType ET, typename InT, typename OutT, typename Functor>
void LaunchSameDimsElementwiseCudaKernel(
    const platform::CUDADeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins,
    std::vector<framework::Tensor *> *outs, Functor func) {
  std::vector<const pten::DenseTensor *> pt_inputs;
  std::vector<pten::DenseTensor *> pt_outputs;
  // TODO(YuanRisheng) *_tmp for cache DenseTensor, because the temporary
  // DenseTensor obj
  // generated by MakePtenDenseTensor can be destroyed when exits loop. *_tmp
  // can be deleted
  // when DenseTensor support copy constructor.
  std::vector<std::unique_ptr<pten::DenseTensor>> pt_inputs_tmp;
  std::vector<std::unique_ptr<pten::DenseTensor>> pt_outputs_tmp;
  for (auto in : ins) {
    pt_inputs_tmp.emplace_back(
        std::move(paddle::experimental::MakePtenDenseTensor(*in)));
  }
  for (auto out : *outs) {
    pt_outputs_tmp.emplace_back(
        std::move(paddle::experimental::MakePtenDenseTensor(*out)));
  }
  for (int i = 0; i < pt_inputs_tmp.size(); i++) {
    pt_inputs.push_back(pt_inputs_tmp[i].get());
  }
  for (int i = 0; i < pt_outputs_tmp.size(); i++) {
    pt_outputs.push_back(pt_outputs_tmp[i].get());
  }
  pten::LaunchSameDimsElementwiseCudaKernel<ET, InT, OutT>(ctx, pt_inputs,
                                                           &pt_outputs, func);
}

}  // namespace operators
}  // namespace paddle
