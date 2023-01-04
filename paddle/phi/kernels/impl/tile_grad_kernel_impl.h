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
#include <type_traits>
#include <vector>

#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/tile_grad_kernel.h"

namespace phi {

template <typename Context, typename T, int Dims>
void TileBackward(const Context& dev_ctx,
                  const DenseTensor& out_grad,
                  const std::vector<int>& reshape_dims_vec,
                  const std::vector<int>& reduce_dims_vec,
                  DenseTensor* x_grad) {
  size_t reshape_size = reshape_dims_vec.size();
  size_t reduce_size = reduce_dims_vec.size();
  dev_ctx.template Alloc<T>(x_grad);

  auto eigen_x_grad = EigenVector<T>::Flatten(*x_grad);
  Eigen::DSizes<Eigen::DenseIndex, Dims * 2> reshape_dims;
  for (size_t i = 0; i < reshape_size; ++i) {
    reshape_dims[i] = reshape_dims_vec[i];
  }
  Eigen::DSizes<Eigen::DenseIndex, Dims> reduce_dims;
  for (size_t i = 0; i < reduce_size; ++i) {
    reduce_dims[i] = reduce_dims_vec[i];
  }

  auto eigen_out_grad = EigenVector<T>::Flatten(out_grad);
  auto& place = *dev_ctx.eigen_device();
  funcs::EigenBroadcastGrad<std::decay_t<decltype(place)>, T, Dims>::Eval(
      place, eigen_x_grad, eigen_out_grad, reduce_dims, reshape_dims);
}

template <typename T, typename Context>
void TileGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out_grad,
                    const IntArray& repeat_times,
                    DenseTensor* x_grad) {
  auto x_dims = x.dims();
  auto vec_x_dims = phi::vectorize<int>(x_dims);
  auto repeat_times_data = repeat_times.GetData();
  if (repeat_times_data.size() < vec_x_dims.size()) {
    int diff = vec_x_dims.size() - repeat_times_data.size();
    repeat_times_data.insert(repeat_times_data.begin(), diff, 1);
  } else {
    int diff = repeat_times_data.size() - vec_x_dims.size();
    vec_x_dims.insert(vec_x_dims.begin(), diff, 1);
  }
  // 1. reshape_dims_vec is the broadcast parameter.
  // 2. reduce_dims_vec is the dimension parameter to compute gradients. For
  //    each dimension expanded, the gradients should be summed to original
  //    size.
  std::vector<int> reshape_dims_vec;
  std::vector<int> reduce_dims_vec;
  for (size_t i = 0; i < repeat_times_data.size(); ++i) {
    reduce_dims_vec.push_back(reshape_dims_vec.size());
    reshape_dims_vec.push_back(repeat_times_data[i]);
    reshape_dims_vec.push_back(vec_x_dims[i]);
  }

  int dims = reduce_dims_vec.size();

  bool just_copy = true;
  for (size_t i = 0; i < repeat_times_data.size(); i++) {
    if (repeat_times_data[i] != 1) {
      just_copy = false;
      break;
    }
  }
  // no need reduce, just copy
  if (just_copy) {
    dev_ctx.template Alloc<T>(x_grad);

    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    // TensorCopy may change the dims of dx
    x_grad->Resize(x_dims);
  } else {
    PADDLE_ENFORCE_GE(dims,
                      1,
                      errors::InvalidArgument(
                          "The rank of the input 'Out@GRAD' for tile_grad op "
                          "must be greater than or equal to 1, but "
                          "the value received is %d.",
                          dims));
    PADDLE_ENFORCE_LE(dims,
                      MAX_RANK_SUPPORTED,
                      errors::InvalidArgument(
                          "The rank of the input 'Out@GRAD' for tile_grad op "
                          "must be less than or equal "
                          "to %d, but the value received is %d.",
                          MAX_RANK_SUPPORTED,
                          dims));
    switch (dims) {
      case 1:
        TileBackward<Context, T, 1>(
            dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, x_grad);
        break;
      case 2:
        TileBackward<Context, T, 2>(
            dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, x_grad);
        break;
      case 3:
        TileBackward<Context, T, 3>(
            dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, x_grad);
        break;
      case 4:
        TileBackward<Context, T, 4>(
            dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, x_grad);
        break;
      case 5:
        TileBackward<Context, T, 5>(
            dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, x_grad);
        break;
      case 6:
        TileBackward<Context, T, 6>(
            dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, x_grad);
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
