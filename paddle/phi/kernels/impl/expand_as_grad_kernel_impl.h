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
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/impl/expand_as_kernel_impl.h"
namespace phi {
template <typename Context, typename T, int Dims>
void ExpandAsBackward(const Context& dev_ctx,
                      const DenseTensor& out_grad,
                      const std::vector<int64_t>& reshape_dims_vec,
                      const std::vector<int>& reduce_dims_vec,
                      DenseTensor* in_grad) {
  size_t reshape_size = reshape_dims_vec.size();
  size_t reduce_size = reduce_dims_vec.size();
  dev_ctx.template Alloc<T>(in_grad);
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
  auto& place = *dev_ctx.eigen_device();
  funcs::EigenBroadcastGrad<std::decay_t<decltype(place)>, T, Dims>::Eval(
      place, x_grad, out_grad0, reduce_dims, reshape_dims);
}

template <typename T, typename Context>
void ExpandAsGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& out_grad,
                        const std::vector<int64_t>& target_shape,
                        DenseTensor* in_grad) {
  if (out_grad.numel() == 0) {
    phi::Full<T, Context>(
        dev_ctx, phi::IntArray(common::vectorize(in_grad->dims())), 0, in_grad);
    return;
  }
  auto x_dims = x.dims();
  auto out_grad_dims = out_grad.dims();
  std::vector<int64_t> real_target_shape =
      phi::vectorize<int64_t>(out_grad_dims);

  if (in_grad->dims() == out_grad_dims) {
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, in_grad);
    return;
  }

  auto vec_in_dims = common::vectorize<int64_t>(x_dims);
  auto diff = real_target_shape.size() - vec_in_dims.size();
  vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  std::vector<int64_t> repeat_times(vec_in_dims.size());
  for (size_t i = 0; i < vec_in_dims.size(); ++i) {
    repeat_times[i] = real_target_shape[i] / vec_in_dims[i];
  }
  std::vector<int64_t> reshape_dims_vec;
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
      errors::InvalidArgument("The rank of the input 'Out@GRAD' for "
                              "expand_as_v2_grad op must be greater than or "
                              "equal to 0, but the value received is %d.",
                              dims));
  PADDLE_ENFORCE_LE(
      dims,
      MAX_RANK_SUPPORTED,
      errors::InvalidArgument("The rank of the input 'Out@GRAD' for "
                              "expand_as_v2_grad op must be less than or equal "
                              "to %d, but the value received is %d.",
                              MAX_RANK_SUPPORTED,
                              dims));
  switch (dims) {
    case 0:
      ExpandAsBackward<Context, T, 0>(
          dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
      break;
    case 1:
      ExpandAsBackward<Context, T, 1>(
          dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
      break;
    case 2:
      ExpandAsBackward<Context, T, 2>(
          dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
      break;
    case 3:
      ExpandAsBackward<Context, T, 3>(
          dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
      break;
    case 4:
      ExpandAsBackward<Context, T, 4>(
          dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
      break;
    case 5:
      ExpandAsBackward<Context, T, 5>(
          dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
      break;
    case 6:
      ExpandAsBackward<Context, T, 6>(
          dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
      break;
    case 7:
      ExpandAsBackward<Context, T, 7>(
          dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
      break;
    case 8:
      ExpandAsBackward<Context, T, 8>(
          dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
      break;
    default:
      PADDLE_THROW(errors::InvalidArgument(
          "Only support tensor with rank being between 1 and %d. But "
          "received tensor's rank = %d.",
          MAX_RANK_SUPPORTED,
          dims));
  }
}

}  // namespace phi
