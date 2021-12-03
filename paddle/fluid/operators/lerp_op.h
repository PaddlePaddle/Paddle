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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
#endif

namespace paddle {
namespace operators {

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

template <typename DeviceContext, typename T, size_t D>
static void LerpFunction(const framework::ExecutionContext& ctx) {
  auto x = ctx.Input<framework::Tensor>("X");
  auto y = ctx.Input<framework::Tensor>("Y");
  auto w = ctx.Input<framework::Tensor>("Weight");
  auto out = ctx.Output<framework::Tensor>("Out");
  out->mutable_data<T>(ctx.GetPlace());

  auto out_dims = out->dims();
  auto x_dims = ExtendDims2Rank(x->dims(), D);
  auto y_dims = ExtendDims2Rank(y->dims(), D);
  auto w_dims = ExtendDims2Rank(w->dims(), D);
  Eigen::DSizes<int, D> x_bcast_dims;
  Eigen::DSizes<int, D> y_bcast_dims;
  Eigen::DSizes<int, D> w_bcast_dims;
  GetBroadcastDims<D>(x_dims, out_dims, &x_bcast_dims);
  GetBroadcastDims<D>(y_dims, out_dims, &y_bcast_dims);
  GetBroadcastDims<D>(w_dims, out_dims, &w_bcast_dims);

  auto eigen_x = framework::EigenTensor<T, D>::From(*x, x_dims);
  auto eigen_y = framework::EigenTensor<T, D>::From(*y, y_dims);
  auto eigen_w = framework::EigenTensor<T, D>::From(*w, w_dims);
  auto eigen_out = framework::EigenTensor<T, D>::From(*out);

  auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
  eigen_out.device(place) =
      eigen_x.broadcast(x_bcast_dims) +
      eigen_w.broadcast(w_bcast_dims) *
          (eigen_y.broadcast(y_bcast_dims) - eigen_x.broadcast(x_bcast_dims));
}

template <typename DeviceContext, typename T, size_t D>
static void LerpGradFunction(const framework::ExecutionContext& ctx) {
  auto w = ctx.Input<framework::Tensor>("Weight");
  auto dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
  auto dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
  auto dy = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));

  auto dout_dims = dout->dims();
  auto dx_dims = ExtendDims2Rank(dx->dims(), D);
  auto dy_dims = ExtendDims2Rank(dy->dims(), D);
  auto w_dims = ExtendDims2Rank(w->dims(), D);
  Eigen::DSizes<int, D> dx_bcast_dims;
  Eigen::DSizes<int, D> dy_bcast_dims;
  Eigen::DSizes<int, D> w_bcast_dims;
  GetBroadcastDims<D>(dx_dims, dout_dims, &dx_bcast_dims);
  GetBroadcastDims<D>(dy_dims, dout_dims, &dy_bcast_dims);
  GetBroadcastDims<D>(w_dims, dout_dims, &w_bcast_dims);

  auto eigen_w = framework::EigenTensor<T, D>::From(*w, w_dims);
  auto eigen_dout = framework::EigenTensor<T, D>::From(*dout);

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

  auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

  if (dx) {
    dx->mutable_data<T>(ctx.GetPlace());
    auto eigen_dx = framework::EigenTensor<T, D>::From(*dx, dx_dims);
    auto eigen_expr = (1 - eigen_w.broadcast(w_bcast_dims)) * eigen_dout;
    eigen_dx.device(place) = eigen_expr.reshape(dx_reshape_dims)
                                 .sum(reduce_dims)
                                 .reshape(eigen_dx.dimensions());
  }
  if (dy) {
    dy->mutable_data<T>(ctx.GetPlace());
    auto eigen_dy = framework::EigenTensor<T, D>::From(*dy, dy_dims);
    auto eigen_expr = eigen_w.broadcast(w_bcast_dims) * eigen_dout;
    eigen_dy.device(place) = eigen_expr.reshape(dy_reshape_dims)
                                 .sum(reduce_dims)
                                 .reshape(eigen_dy.dimensions());
  }
}

template <typename DeviceContext, typename T>
class LerpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int rank = ctx.Output<framework::Tensor>("Out")->dims().size();
    PADDLE_ENFORCE_GE(
        rank, 1,
        platform::errors::InvalidArgument(
            "The number of dimensions for LerpOp must be "
            "greater than or equal to 1, but the value received is %d.",
            rank));
    PADDLE_ENFORCE_LE(
        rank, 6, platform::errors::InvalidArgument(
                     "The number of dimensions for LerpOp must be "
                     "less than or equal to 6, but the value received is %d.",
                     rank));
    switch (rank) {
      case 1:
        LerpFunction<DeviceContext, T, 1>(ctx);
        break;
      case 2:
        LerpFunction<DeviceContext, T, 2>(ctx);
        break;
      case 3:
        LerpFunction<DeviceContext, T, 3>(ctx);
        break;
      case 4:
        LerpFunction<DeviceContext, T, 4>(ctx);
        break;
      case 5:
        LerpFunction<DeviceContext, T, 5>(ctx);
        break;
      case 6:
        LerpFunction<DeviceContext, T, 6>(ctx);
        break;
    }
  }
};

template <typename DeviceContext, typename T>
class LerpGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int rank = ctx.Input<framework::Tensor>(framework::GradVarName("Out"))
                   ->dims()
                   .size();
    PADDLE_ENFORCE_GE(
        rank, 1,
        platform::errors::InvalidArgument(
            "The number of dimensions for LerpGradOp must be "
            "greater than or equal to 1, but the value received is %d.",
            rank));
    PADDLE_ENFORCE_LE(
        rank, 6, platform::errors::InvalidArgument(
                     "The number of dimensions for LerpGradOp must be "
                     "less than or equal to 6, but the value received is %d.",
                     rank));
    switch (rank) {
      case 1:
        LerpGradFunction<DeviceContext, T, 1>(ctx);
        break;
      case 2:
        LerpGradFunction<DeviceContext, T, 2>(ctx);
        break;
      case 3:
        LerpGradFunction<DeviceContext, T, 3>(ctx);
        break;
      case 4:
        LerpGradFunction<DeviceContext, T, 4>(ctx);
        break;
      case 5:
        LerpGradFunction<DeviceContext, T, 5>(ctx);
        break;
      case 6:
        LerpGradFunction<DeviceContext, T, 6>(ctx);
        break;
    }
  }
};

}  // namespace operators
}  // namespace paddle
