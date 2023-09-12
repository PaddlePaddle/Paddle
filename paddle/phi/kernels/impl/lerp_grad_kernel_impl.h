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

#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename Context, typename T, size_t D>
static void LerpGradFunction(const Context& ctx,
                             const DenseTensor& x UNUSED,
                             const DenseTensor& y UNUSED,
                             const DenseTensor& weight,
                             const DenseTensor& out,
                             const DenseTensor& out_grad,
                             DenseTensor* x_grad,
                             DenseTensor* y_grad) {
  auto& w = weight;
  auto& dout = out_grad;
  auto* dx = x_grad;
  auto* dy = y_grad;

  auto& out_dims = out.dims();
  DDim dx_dims;
  DDim dy_dims;

  auto w_dims = phi::funcs::ExtendDims2Rank(w.dims(), D);
  auto g_dims = phi::funcs::ExtendDims2Rank(out_grad.dims(), D);
  Eigen::DSizes<int, D> dx_bcast_dims;
  Eigen::DSizes<int, D> dy_bcast_dims;
  Eigen::DSizes<int, D> w_bcast_dims;
  Eigen::DSizes<int, D> g_bcast_dims;

  if (dx) {
    dx_dims = phi::funcs::ExtendDims2Rank(dx->dims(), D);
    phi::funcs::GetBroadcastDims<D>(dx_dims, out_dims, &dx_bcast_dims);
  }
  if (dy) {
    dy_dims = phi::funcs::ExtendDims2Rank(dy->dims(), D);
    phi::funcs::GetBroadcastDims<D>(dy_dims, out_dims, &dy_bcast_dims);
  }
  phi::funcs::GetBroadcastDims<D>(w_dims, out_dims, &w_bcast_dims);
  phi::funcs::GetBroadcastDims<D>(g_dims, out_dims, &g_bcast_dims);

  auto eigen_w = phi::EigenTensor<T, D>::From(w, w_dims);
  auto eigen_dout = phi::EigenTensor<T, D>::From(dout, g_dims);

  Eigen::DSizes<int, D * 2> dx_reshape_dims;
  Eigen::DSizes<int, D * 2> dy_reshape_dims;
  Eigen::DSizes<int, D> reduce_dims;

  for (int i = 0; i < out_dims.size(); ++i) {
    if (dx) {
      dx_reshape_dims[2 * i] = dx_bcast_dims[i];
      dx_reshape_dims[2 * i + 1] = dx_dims[i];
    }
    if (dy) {
      dy_reshape_dims[2 * i] = dy_bcast_dims[i];
      dy_reshape_dims[2 * i + 1] = dy_dims[i];
    }
    reduce_dims[i] = 2 * i;
  }

  auto& place = *ctx.eigen_device();

  if (dx) {
    ctx.template Alloc<T>(dx);
    auto eigen_dx = phi::EigenTensor<T, D>::From(*dx, dx_dims);
    auto eigen_expr = (1 - eigen_w.broadcast(w_bcast_dims)) *
                      eigen_dout.broadcast(g_bcast_dims);
    eigen_dx.device(place) = eigen_expr.reshape(dx_reshape_dims)
                                 .sum(reduce_dims)
                                 .reshape(eigen_dx.dimensions());
  }
  if (dy) {
    ctx.template Alloc<T>(dy);
    auto eigen_dy = phi::EigenTensor<T, D>::From(*dy, dy_dims);
    auto eigen_expr =
        eigen_w.broadcast(w_bcast_dims) * eigen_dout.broadcast(g_bcast_dims);
    eigen_dy.device(place) = eigen_expr.reshape(dy_reshape_dims)
                                 .sum(reduce_dims)
                                 .reshape(eigen_dy.dimensions());
  }
}

template <typename Context, typename T>
static void LerpGradFunctionZero(const Context& ctx,
                                 const DenseTensor& x UNUSED,
                                 const DenseTensor& y UNUSED,
                                 const DenseTensor& weight,
                                 const DenseTensor& out UNUSED,
                                 const DenseTensor& out_grad,
                                 DenseTensor* x_grad,
                                 DenseTensor* y_grad) {
  auto dim = make_ddim(std::vector<int64_t>(1, 1));
  auto eigen_w = phi::EigenTensor<T, 1>::From(weight, dim);
  auto eigen_dout = phi::EigenTensor<T, 1>::From(out_grad, dim);

  auto& place = *ctx.eigen_device();
  if (x_grad) {
    ctx.template Alloc<T>(x_grad);
    auto eigen_dx = phi::EigenTensor<T, 1>::From(*x_grad, dim);
    eigen_dx.device(place) = (1 - eigen_w) * eigen_dout;
  }
  if (y_grad) {
    ctx.template Alloc<T>(y_grad);
    auto eigen_dy = phi::EigenTensor<T, 1>::From(*y_grad, dim);
    eigen_dy.device(place) = eigen_w * eigen_dout;
  }
}

template <typename T, typename Context>
void LerpGradKernel(const Context& ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    const DenseTensor& weight,
                    const DenseTensor& out,
                    const DenseTensor& out_grad,
                    DenseTensor* x_grad,
                    DenseTensor* y_grad) {
  int rank = out.dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      0,
      phi::errors::InvalidArgument(
          "The number of dimensions for LerpGradOp must be "
          "greater than or equal to 0, but the value received is %d.",
          rank));
  PADDLE_ENFORCE_LE(
      rank,
      6,
      phi::errors::InvalidArgument(
          "The number of dimensions for LerpGradOp must be "
          "less than or equal to 6, but the value received is %d.",
          rank));
  switch (rank) {
    case 0:
      LerpGradFunctionZero<Context, T>(
          ctx, x, y, weight, out, out_grad, x_grad, y_grad);
      break;
    case 1:
      LerpGradFunction<Context, T, 1>(
          ctx, x, y, weight, out, out_grad, x_grad, y_grad);
      break;
    case 2:
      LerpGradFunction<Context, T, 2>(
          ctx, x, y, weight, out, out_grad, x_grad, y_grad);
      break;
    case 3:
      LerpGradFunction<Context, T, 3>(
          ctx, x, y, weight, out, out_grad, x_grad, y_grad);
      break;
    case 4:
      LerpGradFunction<Context, T, 4>(
          ctx, x, y, weight, out, out_grad, x_grad, y_grad);
      break;
    case 5:
      LerpGradFunction<Context, T, 5>(
          ctx, x, y, weight, out, out_grad, x_grad, y_grad);
      break;
    case 6:
      LerpGradFunction<Context, T, 6>(
          ctx, x, y, weight, out, out_grad, x_grad, y_grad);
      break;
  }
}

}  // namespace phi
