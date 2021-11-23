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

namespace paddle {
namespace operators {

static framework::DDim GetNewDims(const framework::DDim& in_dims, int rank) {
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
static void GetBraodcastDims(const framework::DDim& in_dims,
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
  auto out = ctx.Output<framework::Tensor>("Out");
  out->mutable_data<T>(ctx.GetPlace());

  auto out_dims = out->dims();
  auto x_dims = GetNewDims(x->dims(), D);
  auto y_dims = GetNewDims(y->dims(), D);
  Eigen::DSizes<int, D> x_bcast_dims;
  Eigen::DSizes<int, D> y_bcast_dims;
  GetBraodcastDims<D>(x_dims, out_dims, &x_bcast_dims);
  GetBraodcastDims<D>(y_dims, out_dims, &y_bcast_dims);

  auto eigen_x = framework::EigenTensor<T, D>::From(*x, x_dims);
  auto eigen_y = framework::EigenTensor<T, D>::From(*y, y_dims);
  auto eigen_out = framework::EigenTensor<T, D>::From(*out);

  auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
  if (ctx.HasInput("Weight")) {
    auto w = ctx.Input<framework::Tensor>("Weight");
    auto w_dims = GetNewDims(w->dims(), D);
    Eigen::DSizes<int, D> w_bcast_dims;
    GetBraodcastDims<D>(w_dims, out_dims, &w_bcast_dims);
    auto eigen_w = framework::EigenTensor<T, D>::From(*w, w_dims);
    eigen_out.device(place) =
        eigen_x.broadcast(x_bcast_dims) +
        eigen_w.broadcast(w_bcast_dims) *
            (eigen_y.broadcast(y_bcast_dims) - eigen_x.broadcast(x_bcast_dims));
  } else if (ctx.HasAttr("WeightValue")) {
    float w = ctx.Attr<float>("WeightValue");
    eigen_out.device(place) =
        eigen_x.broadcast(x_bcast_dims) +
        w * (eigen_y.broadcast(y_bcast_dims) - eigen_x.broadcast(x_bcast_dims));
  } else {
    PADDLE_THROW(
        platform::errors::InvalidArgument("Must have one of weight or value"));
  }
}

template <typename DeviceContext, typename T, size_t D>
static void LerpGradFunction(const framework::ExecutionContext& ctx) {
  auto x = ctx.Input<framework::Tensor>("X");
  auto y = ctx.Input<framework::Tensor>("Y");
  auto out = ctx.Input<framework::Tensor>("Out");
  auto dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
  auto dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
  auto dy = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));

  auto out_dims = out->dims();
  auto x_dims = GetNewDims(x->dims(), D);
  auto y_dims = GetNewDims(y->dims(), D);
  Eigen::DSizes<int, D> x_bcast_dims;
  Eigen::DSizes<int, D> y_bcast_dims;
  GetBraodcastDims<D>(x_dims, out_dims, &x_bcast_dims);
  GetBraodcastDims<D>(y_dims, out_dims, &y_bcast_dims);

  auto eigen_x = framework::EigenTensor<T, D>::From(*x, x_dims);
  auto eigen_y = framework::EigenTensor<T, D>::From(*y, y_dims);
  auto eigen_out = framework::EigenTensor<T, D>::From(*out);
  auto eigen_dout = framework::EigenTensor<T, D>::From(*dout);

  Eigen::DSizes<int, D * 2> x_reshape_dims;
  Eigen::DSizes<int, D * 2> y_reshape_dims;
  Eigen::DSizes<int, D> reduce_dims;
  for (int i = 0; i < out_dims.size(); ++i) {
    x_reshape_dims[2 * i] = x_bcast_dims[i];
    x_reshape_dims[2 * i + 1] = x_dims[i];
    y_reshape_dims[2 * i] = y_bcast_dims[i];
    y_reshape_dims[2 * i + 1] = y_dims[i];
    reduce_dims[i] = 2 * i;
  }

  auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

  if (ctx.HasInput("Weight")) {
    auto w = ctx.Input<framework::Tensor>("Weight");
    auto w_dims = GetNewDims(w->dims(), D);
    Eigen::DSizes<int, D> w_bcast_dims;
    GetBraodcastDims<D>(w_dims, out_dims, &w_bcast_dims);
    auto eigen_w = framework::EigenTensor<T, D>::From(*w, w_dims);
    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      auto eigen_dx = framework::EigenTensor<T, D>::From(*dx, x_dims);
      auto eigen_expr =
          eigen_dout * (1 +
                        (eigen_out - eigen_y.broadcast(y_bcast_dims)) /
                            (1 - eigen_w).pow(2));
      eigen_dx.device(place) = eigen_expr.reshape(x_reshape_dims)
                                   .sum(reduce_dims)
                                   .reshape(eigen_dx.dimensions());
    }
    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
      auto eigen_dy = framework::EigenTensor<T, D>::From(*dy, y_dims);
      auto eigen_expr =
          eigen_dout *
          (1 + (eigen_out - eigen_x.broadcast(x_bcast_dims)) / eigen_w.pow(2));
      eigen_dy.device(place) = eigen_expr.reshape(y_reshape_dims)
                                   .sum(reduce_dims)
                                   .reshape(eigen_dy.dimensions());
    }
  } else if (ctx.HasAttr("WeightValue")) {
    float w = ctx.Attr<float>("WeightValue");
    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      auto eigen_dx = framework::EigenTensor<T, D>::From(*dx, x_dims);
      auto eigen_expr =
          eigen_dout * (1 +
                        (1 / std::pow(1 - w, 2)) *
                            (eigen_out - eigen_y.broadcast(y_bcast_dims)));
      eigen_dx.device(place) = eigen_expr.reshape(x_reshape_dims)
                                   .sum(reduce_dims)
                                   .reshape(eigen_dx.dimensions());
    }
    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
      auto eigen_dy = framework::EigenTensor<T, D>::From(*dy, y_dims);
      auto eigen_expr =
          eigen_dout * (1 +
                        (1 / std::pow(w, 2)) *
                            (eigen_out - eigen_x.broadcast(x_bcast_dims)));
      eigen_dy.device(place) = eigen_expr.reshape(y_reshape_dims)
                                   .sum(reduce_dims)
                                   .reshape(eigen_dy.dimensions());
    }
  } else {
    PADDLE_THROW(
        platform::errors::InvalidArgument("Must have one of weight or value"));
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
    int rank = ctx.Input<framework::Tensor>("Out")->dims().size();
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
