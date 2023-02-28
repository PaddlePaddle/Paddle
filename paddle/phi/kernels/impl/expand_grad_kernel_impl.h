// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/impl/expand_kernel_impl.h"

namespace phi {
template <typename Context, typename T, int Dims>
void ExpandBackward(const Context& ctx,
                    const DenseTensor& out_grad,
                    const std::vector<int>& reshape_dims_vec,
                    const std::vector<int>& reduce_dims_vec,
                    DenseTensor* in_grad) {
  size_t reshape_size = reshape_dims_vec.size();
  size_t reduce_size = reduce_dims_vec.size();
  ctx.template Alloc<T>(in_grad);
  in_grad->data<T>();

  auto x_grad = EigenVector<T>::Flatten(*in_grad);
  Eigen::DSizes<Eigen::DenseIndex, Dims * 2> reshape_dims;
  for (size_t i = 0; i < reshape_size; ++i) {
    reshape_dims[i] = reshape_dims_vec[i];
  }
  Eigen::DSizes<Eigen::DenseIndex, Dims> reduce_dims;
  for (size_t i = 0; i < reduce_size; ++i) {
    reduce_dims[i] = reduce_dims_vec[i];
  }
  auto out_grad0 = EigenVector<T>::Flatten(out_grad);
  auto& place = *ctx.eigen_device();
  phi::funcs::EigenBroadcastGrad<std::decay_t<decltype(place)>, T, Dims>::Eval(
      place, x_grad, out_grad0, reduce_dims, reshape_dims);
}

template <typename T, typename Context>
void ExpandGradKernel(const Context& ctx,
                      const DenseTensor& x,
                      const DenseTensor& out_grad,
                      const IntArray& shape,
                      DenseTensor* in_grad) {
  auto expand_shape = shape.GetData();
  auto x_dims = x.dims();

  if (in_grad->dims() == out_grad.dims()) {
    phi::Copy(ctx, out_grad, ctx.GetPlace(), false, in_grad);
    return;
  }
  auto vec_in_dims = phi::vectorize<int>(x_dims);
  auto diff = expand_shape.size() - vec_in_dims.size();
  vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  // 1. reshape_dims_vec is the broadcast parameter.
  // 2. reduce_dims_vec is the dimension parameter to compute gradients. For
  //    each dimension expanded, the gradients should be summed to original
  //    size.
  std::vector<int> repeat_times(vec_in_dims.size());
  for (size_t i = 0; i < vec_in_dims.size(); ++i) {
    if (expand_shape[i] < 0) {
      repeat_times[i] = 1;
    } else {
      repeat_times[i] = expand_shape[i] / vec_in_dims[i];
    }
  }
  std::vector<int> reshape_dims_vec;
  std::vector<int> reduce_dims_vec;
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    reduce_dims_vec.push_back(reshape_dims_vec.size());
    reshape_dims_vec.push_back(repeat_times[i]);
    reshape_dims_vec.push_back(vec_in_dims[i]);
  }

  int dims = reduce_dims_vec.size();

  PADDLE_ENFORCE_GE(
      dims,
      0,
      phi::errors::InvalidArgument("The rank of the input 'Out@GRAD' for "
                                   "expand_v2_grad op must be greater than or "
                                   "equal to 0, but the value received is %d.",
                                   dims));
  PADDLE_ENFORCE_LE(dims,
                    MAX_RANK_SUPPORTED,
                    phi::errors::InvalidArgument(
                        "The rank of the input 'Out@GRAD' for "
                        "expand_v2_grad op must be less than or equal "
                        "to %d, but the value received is %d.",
                        MAX_RANK_SUPPORTED,
                        dims));
  switch (dims) {
    case 0:
      ExpandBackward<Context, T, 1>(
          ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
      break;
    case 1:
      ExpandBackward<Context, T, 1>(
          ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
      break;
    case 2:
      ExpandBackward<Context, T, 2>(
          ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
      break;
    case 3:
      ExpandBackward<Context, T, 3>(
          ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
      break;
    case 4:
      ExpandBackward<Context, T, 4>(
          ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
      break;
    case 5:
      ExpandBackward<Context, T, 5>(
          ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
      break;
    case 6:
      ExpandBackward<Context, T, 6>(
          ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
      break;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Only support tensor with rank being between 1 and 6. But "
          "received tensor's rank = %d.",
          dims));
  }
}

}  // namespace phi
