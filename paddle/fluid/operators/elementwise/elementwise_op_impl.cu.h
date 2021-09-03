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

namespace kps = paddle::operators::kernel_primitives;
enum ElementwiseType { kUnary = 1, kBinary = 2, kTernary = 3 };

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
int GetVectorizedSizeForIO(const std::vector<const framework::Tensor *> &ins,
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

template <int VecSize, typename InT, typename OutT, typename Functor>
__global__ void ElementVectorizedUnary(const InT *__restrict__ in0, OutT *out,
                                       int size, Functor func) {
  int data_offset = VecSize * blockIdx.x * blockDim.x;
  // data offset of this block
  int num = size - data_offset;
  num = (VecSize * blockDim.x) > num ? num : VecSize * blockDim.x;
  // the num this time have to deal with
  InT args[VecSize];
  OutT result[VecSize];
  const bool is_reminder = true;
  if (VecSize * blockDim.x > num) {  // reminder segment
    kps::Init<InT, VecSize>(&args[0], static_cast<InT>(1.0f));
    kps::ReadData<InT, VecSize, 1, 1, is_reminder>(args, in0 + data_offset,
                                                   num);
    kps::ElementwiseUnary<InT, OutT, VecSize, 1, 1, Functor>(result, args,
                                                             func);
    kps::WriteData<OutT, VecSize, 1, 1, is_reminder>(out + data_offset, result,
                                                     num);
  } else {  // complete segment
    kps::ReadData<InT, VecSize, 1, 1>(args, in0 + data_offset, num);
    kps::ElementwiseUnary<InT, OutT, VecSize, 1, 1, Functor>(result, args,
                                                             func);
    kps::WriteData<OutT, VecSize, 1, 1>(out + data_offset, result, num);
  }
}

template <int VecSize, typename InT, typename OutT, typename Functor>
__global__ void ElementVectorizedBinary(const InT *__restrict__ in0,
                                        const InT *__restrict__ in1, OutT *out,
                                        int size, Functor func) {
  int data_offset = VecSize * blockIdx.x * blockDim.x;
  // data offset of this block
  int num = size - data_offset;
  num = (VecSize * blockDim.x) > num ? num : VecSize * blockDim.x;
  // the num this time have to deal with
  InT arg0[VecSize];
  InT arg1[VecSize];
  OutT result[VecSize];

  const bool is_reminder = true;
  if (VecSize * blockDim.x > num) {  // reminder segment
    kps::Init<InT, VecSize>(&arg0[0], static_cast<InT>(1.0f));
    kps::Init<InT, VecSize>(&arg1[0], static_cast<InT>(1.0f));
    kps::ReadData<InT, VecSize, 1, 1, is_reminder>(&arg0[0], in0 + data_offset,
                                                   num);
    kps::ReadData<InT, VecSize, 1, 1, is_reminder>(&arg1[0], in1 + data_offset,
                                                   num);
    kps::ElementwiseBinary<InT, OutT, VecSize, 1, 1, Functor>(result, &arg0[0],
                                                              &arg1[0], func);
    kps::WriteData<OutT, VecSize, 1, 1, is_reminder>(out + data_offset, result,
                                                     num);
  } else {  // complete segment
    kps::ReadData<InT, VecSize, 1, 1>(&arg0[0], in0 + data_offset, num);
    kps::ReadData<InT, VecSize, 1, 1>(&arg1[0], in1 + data_offset, num);
    kps::ElementwiseBinary<InT, OutT, VecSize, 1, 1, Functor>(result, &arg0[0],
                                                              &arg1[0], func);
    kps::WriteData<OutT, VecSize, 1, 1>(out + data_offset, result, num);
  }
}

template <int VecSize, typename InT, typename OutT, typename Functor>
__global__ void ElementVectorizedTernary(const InT *__restrict__ in0,
                                         const InT *__restrict__ in1,
                                         const InT *__restrict__ in2, OutT *out,
                                         int size, Functor func) {
  int data_offset = VecSize * blockIdx.x * blockDim.x;
  // data offset of this block
  int num = size - data_offset;
  num = (VecSize * blockDim.x) > num ? num : VecSize * blockDim.x;
  // the num this time have to deal with
  InT args[3][VecSize];
  OutT result[VecSize];

  const bool is_reminder = true;
  if (VecSize * blockDim.x > num) {  // reminder segment
    kps::Init<InT, VecSize>(args[0], static_cast<InT>(1.0f));
    kps::Init<InT, VecSize>(args[1], static_cast<InT>(1.0f));
    kps::Init<InT, VecSize>(args[2], static_cast<InT>(1.0f));
    kps::ReadData<InT, VecSize, 1, 1, is_reminder>(args[0], in0 + data_offset,
                                                   num);
    kps::ReadData<InT, VecSize, 1, 1, is_reminder>(args[1], in1 + data_offset,
                                                   num);
    kps::ReadData<InT, VecSize, 1, 1, is_reminder>(args[2], in2 + data_offset,
                                                   num);
    kps::ElementwiseTernary<InT, OutT, VecSize, 1, 1, Functor>(
        result, args[0], args[1], args[2], func);
    kps::WriteData<OutT, VecSize, 1, 1, is_reminder>(out + data_offset, result,
                                                     num);
  } else {
    kps::ReadData<InT, VecSize, 1, 1>(args[0], in0 + data_offset, num);
    kps::ReadData<InT, VecSize, 1, 1>(args[1], in1 + data_offset, num);
    kps::ReadData<InT, VecSize, 1, 1>(args[2], in2 + data_offset, num);
    kps::ElementwiseTernary<InT, OutT, VecSize, 1, 1, Functor>(
        result, args[0], args[1], args[2], func);
    kps::WriteData<OutT, VecSize, 1, 1>(out + data_offset, result, num);
  }
}

template <ElementwiseType ET, typename InT, typename OutT, typename Functor,
          int VecSize>
void ElementwiseCudaKernel(const platform::CUDADeviceContext &ctx,
                           const std::vector<const framework::Tensor *> &ins,
                           std::vector<framework::Tensor *> *outs,
                           Functor func) {
  auto numel = ins[0]->numel();
  int block_size = GetThreadsConfig(ctx, numel, VecSize);
  int grid_size =
      ((numel + VecSize - 1) / VecSize + block_size - 1) / block_size;
  const InT *in0 = ins[0]->data<InT>();
  OutT *out = (*outs)[0]->data<OutT>();
  // cuda kernel
  auto stream = ctx.stream();
  switch (ET) {
    case ElementwiseType::kTernary:
      ElementVectorizedTernary<VecSize, InT, OutT,
                               Functor><<<grid_size, block_size, 0, stream>>>(
          in0, ins[1]->data<InT>(), ins[2]->data<InT>(), out, numel, func);
      break;
    case ElementwiseType::kBinary:
      ElementVectorizedBinary<VecSize, InT, OutT,
                              Functor><<<grid_size, block_size, 0, stream>>>(
          in0, ins[1]->data<InT>(), out, numel, func);
      break;
    case ElementwiseType::kUnary:
      ElementVectorizedUnary<VecSize, InT, OutT,
                             Functor><<<grid_size, block_size, 0, stream>>>(
          in0, out, numel, func);
      break;
    default: {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported this ElementwiseType : %d !", ET));
      break;
    }
  }
}

template <ElementwiseType ET, typename InT, typename OutT, typename Functor>
void LaunchSameDimsElementwiseCudaKernel(
    const platform::CUDADeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins,
    std::vector<framework::Tensor *> *outs, Functor func) {
  // calculate the max vec_size for all ins and outs
  int vec_size = GetVectorizedSizeForIO<InT, OutT>(ins, *outs);
  switch (vec_size) {
    case 4:
      ElementwiseCudaKernel<ET, InT, OutT, Functor, 4>(ctx, ins, outs, func);
      break;
    case 2:
      ElementwiseCudaKernel<ET, InT, OutT, Functor, 2>(ctx, ins, outs, func);
      break;
    case 1:
      ElementwiseCudaKernel<ET, InT, OutT, Functor, 1>(ctx, ins, outs, func);
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
