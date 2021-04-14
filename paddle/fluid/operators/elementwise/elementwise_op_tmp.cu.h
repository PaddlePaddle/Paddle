// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <algorithm>
#include <utility>
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_tmp.h"

namespace paddle {
namespace operators {

//// template <typename DeviceContext, typename T, typename Functor>
//// class ElementwiseKernel : public framework::OpKernel<T> {
////  public:
////   void Compute(const framework::ExecutionContext &ctx) const override {
////     auto *in_x = ctx.Input<framework::LoDTensor>("X");
////     auto *in_y = ctx.Input<framework::LoDTensor>("Y");
////     auto *out = ctx.Output<framework::LoDTensor>("Out");
//
////     const std::vector< framework::Tensor *> ins;
////     ins.emplace_back(in_x);
////     bool no_broadcast = in_x->dims() == out->dims();;
//
////     if (in_y) {
////       ins.emplace_back(in_y);
////       no_broadcast &= in_y->dims() == out->dims();
////     }
//
////     if (no_broadcast || (ins.size() == 1)) {
////       // SameDimsElemwise<DeviceContext, T, Functor>(
////       //     ctx, &ins, out);
////       ;
////     } else {
////       BroadcastElementwise<DeviceContext, T, Functor>(
////           ctx, &ins, out);
////     }
////   }
//// };

template <typename T, typename loader_t, int N, int nDims>
__global__ void CommonElementwiseKernel(loader_t *in_loaders, T *out_data,
                                        int loop, int remain) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  T args[N];

#pragma unroll
  for (int i = 0; i < N; ++i) {
    in_loaders->load[i](in_loaders->data[i], args[i], loop);
  }

  out_data[tid] = args[0] + args[1];
}

template <typename T, typename offset_t, int N>
void CommonElementwiseCore(const framework::ExecutionContext &ctx,
                           const std::vector<framework::Tensor *> *ins,
                           framework::Tensor *out, offset_t *offset_pre) {
  constexpr int dim_sizes = offset_pre->strides.size();
  T *out_data = out->mutable_data<T>(ctx.GetPlace());

  constexpr int vec_size = 4;
  constexpr int threads = 256;
  int blocks = (out->numel() + threads - 1) / threads;
  int loop = out->numel() / vec_size;
  int remain = out->numel() - loop * vec_size;

  auto in_loader = TensorLoader<T, offset_t, N, 1>(ins, out, offset_pre);
  using loader_t = decltype(in_loader);

  switch (dim_sizes) {
    case 2:
      in_loader = TensorLoader<T, offset_t, N, 2>(ins, out, offset_pre);
      CommonElementwiseKernel<T, loader_t, N, 2><<<blocks, threads>>>(
          &in_loader, out_data, loop, remain);
      break;
    case 3:
      in_loader = TensorLoader<T, offset_t, N, 3>(ins, out, offset_pre);
      CommonElementwiseKernel<T, loader_t, N, 3><<<blocks, threads>>>(
          &in_loader, out_data, loop, remain);
      break;
    case 4:
      in_loader = TensorLoader<T, offset_t, N, 4>(ins, out, offset_pre);
      CommonElementwiseKernel<T, loader_t, N, 4><<<blocks, threads>>>(
          &in_loader, out_data, loop, remain);
      break;
    case 5:
      in_loader = TensorLoader<T, offset_t, N, 5>(ins, out, offset_pre);
      CommonElementwiseKernel<T, loader_t, N, 5><<<blocks, threads>>>(
          &in_loader, out_data, loop, remain);
      break;
    default:
      CommonElementwiseKernel<T, loader_t, N, 1><<<blocks, threads>>>(
          &in_loader, out_data, loop, remain);
      break;
  }
}

template <typename T, typename func_t>
void BroadcastElementwise(const framework::ExecutionContext &ctx,
                          const std::vector<framework::Tensor *> *ins,
                          framework::Tensor *out) {
  auto input_num = ins->size();
  auto merged_dims = DimensionTransform<T>(ins, &(out->dims()), input_num);
  auto offset_pre = OffsetPreCalculator<T, decltype(merged_dims)>(merged_dims);
  using offset_t = decltype(offset_pre);

  switch (input_num) {
    case 2: {
      CommonElementwiseCore<T, offset_t, 2>(ctx, ins, out, offset_pre);
      break;
    }
    case 3: {
      CommonElementwiseCore<T, offset_t, 3>(ctx, ins, out, offset_pre);
      break;
    }
    default: { ; }
  }
}

}  // namespace operators
}  // namespace paddle
