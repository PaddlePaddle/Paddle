// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

#define MAX_RANK_SUPPORTED 8

namespace phi {

template <typename T, typename Context, int Rank>
void Expand(const Context& dev_ctx,
            const DenseTensor& x_in,
            const IntArray& shape,
            DenseTensor* out) {
  auto* in0 = &x_in;
  auto in_dims = in0->dims();
  auto expand_times = shape.GetData();
  PADDLE_ENFORCE_EQ(static_cast<size_t>(in_dims.size()),
                    expand_times.size(),
                    common::errors::InvalidArgument(
                        "The number of elements (%d) of 'expand_times' for "
                        "Op(expand) must be equal to the number "
                        "of dimensions (%d) of the input.",
                        expand_times.size(),
                        static_cast<size_t>(in_dims.size())));
  auto* out0 = out;
  Eigen::DSizes<Eigen::DenseIndex, Rank> bcast_dims;
  for (size_t i = 0; i < expand_times.size(); ++i) {
    bcast_dims[i] = expand_times[i];
  }

  phi::DDim out_dims(in_dims);
  for (size_t i = 0; i < expand_times.size(); ++i) {
    out_dims[i] *= expand_times[i];
  }

  out0->Resize(out_dims);
  auto x = EigenTensor<T, Rank>::From(*in0);
  dev_ctx.template Alloc<T>(out0);
  auto y = EigenTensor<T, Rank>::From(*out0);
  auto& place = *dev_ctx.eigen_device();
  // use 32-bit index to speed up
  bool use_32bit_index = y.size() < Eigen::NumTraits<int>::highest();
  if (use_32bit_index) {
    phi::funcs::EigenBroadcast<std::decay_t<decltype(place)>, T, Rank>::Eval(
        place, To32BitIndex(y), To32BitIndex(x), bcast_dims);
  } else {
    phi::funcs::EigenBroadcast<std::decay_t<decltype(place)>, T, Rank>::Eval(
        place, y, x, bcast_dims);
  }
}

template <typename T, typename Context>
void LegacyExpandKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const IntArray& shape,
                        DenseTensor* out) {
  auto rank = x.dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      1,
      common::errors::InvalidArgument(
          "The number of dimensions of the input 'x' for Op(expand) "
          "must be greater than or equal to 1, but the value received is %d.",
          rank));
  PADDLE_ENFORCE_LE(
      rank,
      MAX_RANK_SUPPORTED,
      common::errors::InvalidArgument(
          "The number of dimensions of the input 'x' for Op(expand) "
          "must be less than or equal to %d, but the value received is %d.",
          MAX_RANK_SUPPORTED,
          rank));
  switch (rank) {
    case 1:
      Expand<T, Context, 1>(dev_ctx, x, shape, out);
      break;
    case 2:
      Expand<T, Context, 2>(dev_ctx, x, shape, out);
      break;
    case 3:
      Expand<T, Context, 3>(dev_ctx, x, shape, out);
      break;
    case 4:
      Expand<T, Context, 4>(dev_ctx, x, shape, out);
      break;
    case 5:
      Expand<T, Context, 5>(dev_ctx, x, shape, out);
      break;
    case 6:
      Expand<T, Context, 6>(dev_ctx, x, shape, out);
      break;
    case 7:
      Expand<T, Context, 7>(dev_ctx, x, shape, out);
      break;
    case 8:
      Expand<T, Context, 8>(dev_ctx, x, shape, out);
      break;
  }
}

template <typename T, typename Context, int Dims>
void ExpandBackward(const Context& dev_ctx,
                    const DenseTensor& out_grad_in,
                    const std::vector<int>& reshape_dims_vec,
                    const std::vector<int>& reduce_dims_vec,
                    DenseTensor* in_grad) {
  size_t reshape_size = reshape_dims_vec.size();
  size_t reduce_size = reduce_dims_vec.size();
  PADDLE_ENFORCE_EQ(reshape_size,
                    reshape_dims_vec.size(),
                    common::errors::InvalidArgument(
                        "Inconsistent size between template Dims (%d) and "
                        "reshape dimensions (%d).",
                        reshape_size,
                        reshape_dims_vec.size()));
  PADDLE_ENFORCE_EQ(reduce_size,
                    reduce_dims_vec.size(),
                    common::errors::InvalidArgument(
                        "Inconsistent size between template Dims (%d) and "
                        "reduce dimensions (%d).",
                        reduce_size,
                        reduce_dims_vec.size()));
  auto* in0 = &out_grad_in;
  auto* out0 = in_grad;
  dev_ctx.template Alloc<T>(out0);
  auto x_grad = EigenVector<T>::Flatten(*out0);
  Eigen::DSizes<Eigen::DenseIndex, Dims * 2> reshape_dims;
  for (size_t i = 0; i < reshape_size; ++i) {
    reshape_dims[i] = reshape_dims_vec[i];
  }
  Eigen::DSizes<Eigen::DenseIndex, Dims> reduce_dims;
  for (size_t i = 0; i < reduce_size; ++i) {
    reduce_dims[i] = reduce_dims_vec[i];
  }
  auto out_grad = EigenVector<T>::Flatten(*in0);
  auto& place = *dev_ctx.eigen_device();
  phi::funcs::EigenBroadcastGrad<std::decay_t<decltype(place)>, T, Dims>::Eval(
      place, x_grad, out_grad, reduce_dims, reshape_dims);
}

template <typename T, typename Context>
void LegacyExpandGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& out_grad,
                            const IntArray& shape,
                            DenseTensor* in_grad) {
  auto* in0 = &x;
  // auto& expand_times = context.Attr<std::vector<int>>("expand_times");
  auto expand_times = shape.GetData();
  auto x_dims = in0->dims();
  // 1. reshape_dims_vec is the broadcast parameter.
  // 2. reduce_dims_vec is the dimension parameter to compute gradients. For
  //    each dimension expanded, the gradients should be summed to original
  //    size.
  std::vector<int> reshape_dims_vec;
  std::vector<int> reduce_dims_vec;
  for (size_t i = 0; i < expand_times.size(); ++i) {
    reduce_dims_vec.push_back(reshape_dims_vec.size());
    reshape_dims_vec.push_back(expand_times[i]);
    reshape_dims_vec.push_back(x_dims[i]);
  }

  int dims = reduce_dims_vec.size();

  bool just_copy = true;
  for (size_t i = 0; i < expand_times.size(); i++) {
    if (expand_times[i] != 1) {
      just_copy = false;
      break;
    }
  }
  // no need reduce, just copy
  if (just_copy) {
    auto* in0 = &out_grad;
    auto* out0 = in_grad;
    dev_ctx.template Alloc<T>(out0);
    phi::Copy(dev_ctx, *in0, dev_ctx.GetPlace(), false, out0);
  } else {
    PADDLE_ENFORCE_GE(dims,
                      1,
                      common::errors::InvalidArgument(
                          "The number of dimensions of the input "
                          "'Out@GRAD' for Op(expand_grad)"
                          " must be greater than or equal to 1, but "
                          "the value received is %d.",
                          dims));
    PADDLE_ENFORCE_LE(dims,
                      MAX_RANK_SUPPORTED,
                      common::errors::InvalidArgument(
                          "The number of dimensions of the input 'Out@GRAD' "
                          "for Op(expand_grad) must be less than or equal "
                          "to %d, but the value received is %d.",
                          MAX_RANK_SUPPORTED,
                          dims));
    switch (dims) {
      case 1:
        ExpandBackward<T, Context, 1>(
            dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
        break;
      case 2:
        ExpandBackward<T, Context, 2>(
            dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
        break;
      case 3:
        ExpandBackward<T, Context, 3>(
            dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
        break;
      case 4:
        ExpandBackward<T, Context, 4>(
            dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
        break;
      case 5:
        ExpandBackward<T, Context, 5>(
            dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
        break;
      case 6:
        ExpandBackward<T, Context, 6>(
            dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
        break;
      case 7:
        ExpandBackward<T, Context, 7>(
            dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
        break;
      case 8:
        ExpandBackward<T, Context, 8>(
            dev_ctx, out_grad, reshape_dims_vec, reduce_dims_vec, in_grad);
        break;
      default:
        PADDLE_THROW(common::errors::InvalidArgument(
            "Only support tensor with rank being between 1 and %d. But "
            "received tensor's rank = %d.",
            MAX_RANK_SUPPORTED,
            dims));
    }
  }
}
}  // namespace phi
