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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename Context, typename T, size_t D>
static void LerpFunction(const Context& ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         const DenseTensor& weight,
                         DenseTensor* out) {
  ctx.template Alloc<T>(out);

  auto out_dims = out->dims();
  auto x_dims = phi::funcs::ExtendDims2Rank(x.dims(), D);
  auto y_dims = phi::funcs::ExtendDims2Rank(y.dims(), D);
  auto w_dims = phi::funcs::ExtendDims2Rank(weight.dims(), D);
  Eigen::DSizes<int, D> x_bcast_dims;
  Eigen::DSizes<int, D> y_bcast_dims;
  Eigen::DSizes<int, D> w_bcast_dims;
  phi::funcs::GetBroadcastDims<D>(x_dims, out_dims, &x_bcast_dims);
  phi::funcs::GetBroadcastDims<D>(y_dims, out_dims, &y_bcast_dims);
  phi::funcs::GetBroadcastDims<D>(w_dims, out_dims, &w_bcast_dims);

  auto eigen_x = phi::EigenTensor<T, D>::From(x, x_dims);
  auto eigen_y = phi::EigenTensor<T, D>::From(y, y_dims);
  auto eigen_w = phi::EigenTensor<T, D>::From(weight, w_dims);
  auto eigen_out = phi::EigenTensor<T, D>::From(*out);

  auto& place = *ctx.eigen_device();
  eigen_out.device(place) =
      eigen_x.broadcast(x_bcast_dims) +
      eigen_w.broadcast(w_bcast_dims) *
          (eigen_y.broadcast(y_bcast_dims) - eigen_x.broadcast(x_bcast_dims));
}

template <typename T, typename Context>
void LerpKernel(const Context& ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                const DenseTensor& weight,
                DenseTensor* out) {
  int rank = out->dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      1,
      phi::errors::InvalidArgument(
          "The number of dimensions for LerpOp must be "
          "greater than or equal to 1, but the value received is %d.",
          rank));
  PADDLE_ENFORCE_LE(
      rank,
      6,
      phi::errors::InvalidArgument(
          "The number of dimensions for LerpOp must be "
          "less than or equal to 6, but the value received is %d.",
          rank));
  switch (rank) {
    case 1:
      LerpFunction<Context, T, 1>(ctx, x, y, weight, out);
      break;
    case 2:
      LerpFunction<Context, T, 2>(ctx, x, y, weight, out);
      break;
    case 3:
      LerpFunction<Context, T, 3>(ctx, x, y, weight, out);
      break;
    case 4:
      LerpFunction<Context, T, 4>(ctx, x, y, weight, out);
      break;
    case 5:
      LerpFunction<Context, T, 5>(ctx, x, y, weight, out);
      break;
    case 6:
      LerpFunction<Context, T, 6>(ctx, x, y, weight, out);
      break;
  }
}

}  // namespace phi
