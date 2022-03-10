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

#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/impl/expand_as_kernel_impl.h"

namespace phi {
template <typename Context, typename T, int Dims>
void ExpandAsBackward(const Context& ctx,
                      const DenseTensor& out_grad,
                      const std::vector<int>& reshape_dims_vec,
                      const std::vector<int>& reduce_dims_vec,
                      DenseTensor* in_grad) {
  size_t reshape_size = reshape_dims_vec.size();
  size_t reduce_size = reduce_dims_vec.size();
  ctx.template Alloc<T>(in_grad);
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
  funcs::EigenBroadcastGrad<std::decay_t<decltype(place)>, T, Dims>::Eval(
      place, x_grad, out_grad0, reduce_dims, reshape_dims);
}

template <typename T, typename Context>
void ExpandAsGradKernel(const Context& context,
                        const DenseTensor& x,
                        const DenseTensor& out_grad,
                        const std::vector<int>& target_shape,
                        DenseTensor* in_grad) {
  auto x_dims = x.dims();
  auto vec_in_dims = phi::vectorize<int>(x_dims);
  auto diff = target_shape.size() - vec_in_dims.size();
  vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  std::vector<int> repeat_times(vec_in_dims.size());
  for (size_t i = 0; i < vec_in_dims.size(); ++i) {
    repeat_times[i] = target_shape[i] / vec_in_dims[i];
  }
  std::vector<int> reshape_dims_vec;
  std::vector<int> reduce_dims_vec;
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    reduce_dims_vec.push_back(reshape_dims_vec.size());
    reshape_dims_vec.push_back(repeat_times[i]);
    reshape_dims_vec.push_back(vec_in_dims[i]);
  }

  int dims = reduce_dims_vec.size();
  bool just_copy = true;
  for (size_t i = 0; i < repeat_times.size(); i++) {
    if (repeat_times[i] != 1) {
      just_copy = false;
      break;
    }
  }
  // no need reduce, just copy
  if (just_copy) {
    context.template Alloc<T>(in_grad);
    phi::Copy(context, out_grad, context.GetPlace(), false, in_grad);
  } else {
    PADDLE_ENFORCE_GE(
        dims,
        1,
        errors::InvalidArgument("The rank of the input 'Out@GRAD' for "
                                "expand_as_v2_grad op must be greater than or "
                                "equal to 1, but the value received is %d.",
                                dims));
    PADDLE_ENFORCE_LE(dims,
                      MAX_RANK_SUPPORTED,
                      errors::InvalidArgument(
                          "The rank of the input 'Out@GRAD' for "
                          "expand_as_v2_grad op must be less than or equal "
                          "to %d, but the value received is %d.",
                          MAX_RANK_SUPPORTED,
                          dims));
    switch (dims) {
      case 1:
        ExpandAsBackward<Context, T, 1>(
            context, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
        break;
      case 2:
        ExpandAsBackward<Context, T, 2>(
            context, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
        break;
      case 3:
        ExpandAsBackward<Context, T, 3>(
            context, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
        break;
      case 4:
        ExpandAsBackward<Context, T, 4>(
            context, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
        break;
      case 5:
        ExpandAsBackward<Context, T, 5>(
            context, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
        break;
      case 6:
        ExpandAsBackward<Context, T, 6>(
            context, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
        break;
      default:
        PADDLE_THROW(errors::InvalidArgument(
            "Only support tensor with rank being between 1 and 6. But "
            "received tensor's rank = %d.",
            dims));
    }
  }
}

}  // namespace phi
