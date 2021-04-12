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
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_tmp.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename T, int vec_size>
struct alignas(sizeof(T) * 4) aligned_vector {
  T scalar_array[4];
};

template <typename DeviceContext, typename T, typename Functor>
class ElementwiseKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto in_tensors = ctx.MultiInput<framework::LoDTensor>("Input");
    auto *out_tensor = ctx.Output<framework::LoDTensor>("Out");
    bool no_broadcast = true;

    using vec_input_t = std::vector<const framework::Tensor *>;
    vec_input_t in_tensor_arr;
    in_tensor_arr.reserve(in_tensors.size());
    for (auto *in_tensor : in_tensors) {
      no_broadcast &= in_tensor->dims() == out_tensor->dims();
      in_tensor_arr.emplace_back(in_tensor);
    }
    if (no_broadcast || (in_tensors.size() == 1)) {
      SameDimsElemwise<DeviceContext, T, Functor, vec_input_t>(
          ctx, in_tensor_arr, out_tensor);
    } else {
      BroadcastElementwise<DeviceContext, T, Functor, vec_input_t>(
          ctx, in_tensor_arr, out_tensor);
    }
  }
};

template <typename T, typename Functor, typename vec_input_t>
void BroadcastElementwise(const framework::ExecutionContext &ctx,
                          vec_input_t *ins, framework::Tensor *out) {
  auto input_num = ins->size();
  auto merged_dims = MergeDims<T, vec_input_t>(ins, &(out->dims()), input_num);
  auto offset_pre_cal =
      OffsetPreCalculator<T, decltype(merged_dims)>(merged_dims);

  switch (input_num) {
    case 2: {
      CommonElementwiseCore<T, Functor, decltype(offset_pre_cal), 2>(
          ins, out, offset_pre_cal);
      break;
    }
    case 3: {
      CommonElementwiseCore<T, Functor, decltype(offset_pre_cal), 3>(
          ins, out, offset_pre_cal);
      break;
    }
    default: { ; }
  }
}

template <typename T, typename func_t, typename offset_pre_t, int N>
void CommonElementwiseCore(vec_input_t *ins, framework::Tensor *out,
                           offset_pre_t offset_pre_cal) {
  constexpr int dim_sizes = offset_pre_cal.strides.size();
  auto in_loader = TensorLoader<vec_input_t, decltype(offset_pre_cal), N, 1>;
  switch (dim_sizes) {
    case 2:
      in_loader = TensorLoader<vec_input_t, decltype(offset_pre_cal), N, 2>(
          ins, offset_pre_cal);
      CommonElementwiseKernel<T, decltype(in_loader), func_t, N, 2>(in_loader,
                                                                    out);
      break;
    case 3:
      in_loader = TensorLoader<vec_input_t, decltype(offset_pre_cal), N, 3>(
          ins, offset_pre_cal);
      CommonElementwiseKernel<T, decltype(in_loader), func_t, N, 3>(in_loader,
                                                                    out);
      break;
    case 4:
      in_loader = TensorLoader<vec_input_t, decltype(offset_pre_cal), N, 4>(
          ins, offset_pre_cal);
      CommonElementwiseKernel<T, decltype(in_loader), func_t, N, 4>(in_loader,
                                                                    out);
      break;
    case 5:
      in_loader = TensorLoader<vec_input_t, decltype(offset_pre_cal), N, 5>(
          ins, offset_pre_cal);
      CommonElementwiseKernel<T, decltype(in_loader), func_t, N, 4>(in_loader,
                                                                    out);
      break;
    default:
      CommonElementwiseKernel<T, decltype(in_loader), func_t, N, 1>(in_loader,
                                                                    out);
      break;
  }
}

template <typename T, typename loader_t, typename func_t, int N, int nDims>
__devide__ void CommonElementwiseKernel(vec_data *in_data_arr[], out_data) {
  // T *args[N];

  // #pragma unroll
  // for (int i = 0; i < N; ++i) {
  //   (loader[i])(in_data_arr[i], args[i]);
  // }
  // args[N - 1] = Functor(args);
}

}  // namespace operators
}  // namespace paddle
