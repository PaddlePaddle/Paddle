// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.1
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/fluid/operators/kernel_primitives/kernel_primitives.h"

namespace paddle {
namespace operators {

template <ElementwiseType ET, typename InT, typename OutT, typename Functor>
void LaunchBroadcastElementwiseCudaKernel(
    const platform::CUDADeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins,
    std::vector<framework::Tensor *> *outs, int axis, Functor func) {
  using Traits = platform::FunctionTraits<Functor>;
  const int kArity =
      Traits::has_pointer_args ? static_cast<int>(ET) : Traits::arity;
  PADDLE_ENFORCE_EQ(ins.size(), kArity,
                    platform::errors::InvalidArgument(
                        "The number of inputs is expected to be equal to the "
                        "arity of functor. But recieved: the number of inputs "
                        "is %d, the arity of functor is %d.",
                        ins.size(), kArity));
  PADDLE_ENFORCE_EQ(kArity, 2,
                    platform::errors::InvalidArgument(
                        "Currently only broadcast of binary is supported and "
                        "verified, but received %d.",
                        kArity));

  framework::Tensor *out = (*outs)[0];
  int numel = out->numel();
  const auto merge_dims = DimensionsTransform(ins, out->dims(), axis);
  framework::Array<bool, ET> use_broadcast;
  for (int i = 0; i < ET; i++) {
    use_broadcast[i] = (ins[i]->numel() != numel);
  }
#define CALL_BROADCAST_FOR_DIM_SIZE(rank)                      \
  case rank: {                                                 \
    LaunchElementwiseKernel<InT, OutT, Functor, kArity, rank>( \
        ctx, ins, out, func, merge_dims, use_broadcast);       \
  } break;

  switch (merge_dims.dim_size) {
    CALL_BROADCAST_FOR_DIM_SIZE(1);
    CALL_BROADCAST_FOR_DIM_SIZE(2);
    CALL_BROADCAST_FOR_DIM_SIZE(3);
    CALL_BROADCAST_FOR_DIM_SIZE(4);
    CALL_BROADCAST_FOR_DIM_SIZE(5);
    CALL_BROADCAST_FOR_DIM_SIZE(6);
    CALL_BROADCAST_FOR_DIM_SIZE(7);
    CALL_BROADCAST_FOR_DIM_SIZE(8);
    default: {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The maximum dimension of input tensor is expected to be less than "
          "%d, but recieved %d.\n",
          merge_dims.dim_size, framework::DDim::kMaxRank));
    }
  }
#undef CALL_BROADCAST_FOR_DIM_SIZE
}

template <ElementwiseType ET, typename InT, typename OutT, typename Functor>
void LaunchElementwiseCudaKernel(
    const platform::CUDADeviceContext &cuda_ctx,
    const std::vector<const framework::Tensor *> &ins,
    std::vector<framework::Tensor *> *outs, int axis, Functor func) {
  std::vector<int> dims_size;
  bool no_broadcast_flag = true;
  for (auto *in : ins) {
    no_broadcast_flag = ins[0]->dims() == in->dims();
    dims_size.emplace_back(in->dims().size());
  }

  if (no_broadcast_flag) {
    LaunchSameDimsElementwiseCudaKernel<ET, InT, OutT>(cuda_ctx, ins, outs,
                                                       func);
  } else {
    axis = axis == -1
               ? *std::max_element(dims_size.begin(), dims_size.end()) -
                     *std::min_element(dims_size.begin(), dims_size.end())
               : axis;
    LaunchBroadcastElementwiseCudaKernel<ET, InT, OutT>(cuda_ctx, ins, outs,
                                                        axis, func);
  }
}

}  // namespace operators
}  // namespace paddle
