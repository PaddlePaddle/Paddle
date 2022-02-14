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

#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/kernels/funcs/eigen/common.h"
#include "paddle/pten/kernels/lerp_grad_kernel.h"

namespace pten {

static framework::DDim ExtendDims2Rank(const framework::DDim& in_dims,
                                       int rank) {
  if (in_dims.size() == rank) {
    return in_dims;
  }
  std::vector<int64_t> shapes(rank, 1);
  for (int i = in_dims.size() - 1, j = rank - 1; i >= 0; --i, --j) {
    shapes[j] = in_dims[i];
  }
  return framework::make_ddim(shapes);
}

template <size_t D>
static void GetBroadcastDims(const framework::DDim& in_dims,
                             const framework::DDim& out_dims,
                             Eigen::DSizes<int, D>* bcast_dims) {
  for (size_t i = 0; i < D; ++i) {
    if (in_dims[i] == out_dims[i]) {
      (*bcast_dims)[i] = 1;
    } else {
      (*bcast_dims)[i] = std::max(in_dims[i], out_dims[i]);
    }
  }
}

template <typename Context, typename T, size_t D>
static void LerpGradFunction(const Context& ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             const DenseTensor& weight,
                             const DenseTensor& out,
                             const DenseTensor& out_grad,
                             DenseTensor* x_grad,
                             DenseTensor* y_grad) {
  // auto w = ctx.Input<framework::Tensor>("Weight");
  // auto dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
  // auto dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
  // auto dy = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));

  auto& w = weight;
  auto& dout = out_grad;
  auto* dx = x_grad;
  auto* dy = y_grad;

  auto dout_dims = dout.dims();
  auto dx_dims = ExtendDims2Rank(dx->dims(), D);
  auto dy_dims = ExtendDims2Rank(dy->dims(), D);
  auto w_dims = ExtendDims2Rank(w.dims(), D);
  Eigen::DSizes<int, D> dx_bcast_dims;
  Eigen::DSizes<int, D> dy_bcast_dims;
  Eigen::DSizes<int, D> w_bcast_dims;
  GetBroadcastDims<D>(dx_dims, dout_dims, &dx_bcast_dims);
  GetBroadcastDims<D>(dy_dims, dout_dims, &dy_bcast_dims);
  GetBroadcastDims<D>(w_dims, dout_dims, &w_bcast_dims);

  auto eigen_w = pten::EigenTensor<T, D>::From(w, w_dims);
  auto eigen_dout = pten::EigenTensor<T, D>::From(dout);

  Eigen::DSizes<int, D * 2> dx_reshape_dims;
  Eigen::DSizes<int, D * 2> dy_reshape_dims;
  Eigen::DSizes<int, D> reduce_dims;
  for (int i = 0; i < dout_dims.size(); ++i) {
    dx_reshape_dims[2 * i] = dx_bcast_dims[i];
    dx_reshape_dims[2 * i + 1] = dx_dims[i];
    dy_reshape_dims[2 * i] = dy_bcast_dims[i];
    dy_reshape_dims[2 * i + 1] = dy_dims[i];
    reduce_dims[i] = 2 * i;
  }

  // auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
  auto& place = *ctx.eigen_device();

  if (dx) {
    dx->mutable_data<T>(ctx.GetPlace());
    auto eigen_dx = pten::EigenTensor<T, D>::From(*dx, dx_dims);
    auto eigen_expr = (1 - eigen_w.broadcast(w_bcast_dims)) * eigen_dout;
    eigen_dx.device(place) = eigen_expr.reshape(dx_reshape_dims)
                                 .sum(reduce_dims)
                                 .reshape(eigen_dx.dimensions());
  }
  if (dy) {
    dy->mutable_data<T>(ctx.GetPlace());
    auto eigen_dy = pten::EigenTensor<T, D>::From(*dy, dy_dims);
    auto eigen_expr = eigen_w.broadcast(w_bcast_dims) * eigen_dout;
    eigen_dy.device(place) = eigen_expr.reshape(dy_reshape_dims)
                                 .sum(reduce_dims)
                                 .reshape(eigen_dy.dimensions());
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
      1,
      pten::errors::InvalidArgument(
          "The number of dimensions for LerpGradOp must be "
          "greater than or equal to 1, but the value received is %d.",
          rank));
  PADDLE_ENFORCE_LE(
      rank,
      6,
      pten::errors::InvalidArgument(
          "The number of dimensions for LerpGradOp must be "
          "less than or equal to 6, but the value received is %d.",
          rank));
  switch (rank) {
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

}  // namespace pten
