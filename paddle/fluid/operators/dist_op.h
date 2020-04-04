// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <math.h>
#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
using framework::Tensor;

template <int Rank>
static void GetBraodcastDims(const framework::DDim& x_dims,
                             const framework::DDim& y_dims,
                             Eigen::DSizes<int, Rank>* x_bcast_dims,
                             Eigen::DSizes<int, Rank>* y_bcast_dims) {
  int bcast_dims_remainder = 0;
  for (int i = 0; i < x_dims.size(); ++i) {
    if (x_dims[i] >= y_dims[i]) {
      (*x_bcast_dims)[i] = 1;
      (*y_bcast_dims)[i] = x_dims[i] / y_dims[i];
      bcast_dims_remainder += x_dims[i] % y_dims[i];
    } else {
      (*y_bcast_dims)[i] = 1;
      (*x_bcast_dims)[i] = y_dims[i] / x_dims[i];
      bcast_dims_remainder += y_dims[i] % x_dims[i];
    }
  }
  PADDLE_ENFORCE_EQ(bcast_dims_remainder, 0,
                    "The input tensor of Op(dist) could not be broadcast");
}

static framework::DDim GetNewDims(const framework::DDim& in_dims, int rank) {
  std::vector<int64_t> new_dims_vec(rank);
  if (in_dims.size() < rank) {
    for (int i = 0; i < rank - in_dims.size(); ++i) {
      new_dims_vec[i] = 1;
    }
    for (int i = 0; i < in_dims.size(); ++i) {
      new_dims_vec[i + rank - in_dims.size()] = in_dims[i];
    }
  } else {
    new_dims_vec = vectorize(in_dims);
  }
  return framework::make_ddim(new_dims_vec);
}

template <int Rank>
static void GetReshapeAndReduceDims(
    const framework::DDim& x_dims, const framework::DDim& y_dims,
    const Eigen::DSizes<int, Rank>& x_bcast_dims,
    const Eigen::DSizes<int, Rank>& y_bcast_dims,
    Eigen::DSizes<int, Rank * 2>* x_reshape_dims,
    Eigen::DSizes<int, Rank * 2>* y_reshape_dims,
    Eigen::DSizes<int, Rank>* reduce_dims) {
  for (int i = 0; i < x_dims.size(); ++i) {
    (*x_reshape_dims)[2 * i] = x_bcast_dims[i];
    (*x_reshape_dims)[2 * i + 1] = x_dims[i];
    (*y_reshape_dims)[2 * i] = y_bcast_dims[i];
    (*y_reshape_dims)[2 * i + 1] = y_dims[i];
    (*reduce_dims)[i] = 2 * i;
  }
}

template <typename DeviceContext, typename T, int Rank>
static void DistFunction(const framework::ExecutionContext& context) {
  auto* x = context.Input<Tensor>("X");
  auto* y = context.Input<Tensor>("Y");
  auto* out = context.Output<Tensor>("Out");
  auto p = context.Attr<float>("p");
  out->mutable_data<T>(context.GetPlace());

  auto x_dims = context.Input<Tensor>("X")->dims();
  auto y_dims = context.Input<Tensor>("Y")->dims();

  // first: make new dims with same size as rank
  framework::DDim x_new_dims = GetNewDims(x_dims, Rank);
  framework::DDim y_new_dims = GetNewDims(y_dims, Rank);

  auto x_t = EigenTensor<T, Rank>::From(*x, x_new_dims);
  auto y_t = EigenTensor<T, Rank>::From(*y, y_new_dims);
  auto out_t = EigenTensor<T, 1>::From(*out);
  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();

  Eigen::DSizes<int, Rank> x_bcast_dims;
  Eigen::DSizes<int, Rank> y_bcast_dims;
  GetBraodcastDims<Rank>(x_new_dims, y_new_dims, &x_bcast_dims, &y_bcast_dims);
  if (p == 0) {
    out_t.device(place) =
        (x_t.broadcast(x_bcast_dims) != y_t.broadcast(y_bcast_dims))
            .template cast<T>()
            .sum();
  } else if (p == INFINITY) {
    out_t.device(place) =
        (x_t.broadcast(x_bcast_dims) - y_t.broadcast(y_bcast_dims))
            .abs()
            .maximum();
  } else if (p == -INFINITY) {
    out_t.device(place) =
        (x_t.broadcast(x_bcast_dims) - y_t.broadcast(y_bcast_dims))
            .abs()
            .minimum();
  } else {
    out_t.device(place) =
        (x_t.broadcast(x_bcast_dims) - y_t.broadcast(y_bcast_dims))
            .abs()
            .pow(p)
            .sum()
            .pow(1.0 / p);
  }
}

template <typename DeviceContext, typename T, int Rank>
static void DistGradFunction(const framework::ExecutionContext& context) {
  auto* x = context.Input<Tensor>("X");
  auto* y = context.Input<Tensor>("Y");
  auto* out = context.Input<Tensor>("Out");
  auto p = context.Attr<float>("p");

  auto x_grad = context.Output<Tensor>(framework::GradVarName("X"));
  auto y_grad = context.Output<Tensor>(framework::GradVarName("Y"));
  auto out_grad = context.Input<Tensor>(framework::GradVarName("Out"));

  auto x_dims = context.Input<Tensor>("X")->dims();
  auto y_dims = context.Input<Tensor>("Y")->dims();
  auto out_dims = context.Input<Tensor>("Out")->dims();

  // first: make new dims with same size as rank
  framework::DDim x_new_dims = GetNewDims(x_dims, Rank);
  framework::DDim y_new_dims = GetNewDims(y_dims, Rank);
  framework::DDim out_new_dims = GetNewDims(out_dims, Rank);
  auto x_t = EigenTensor<T, Rank>::From(*x, x_new_dims);
  auto y_t = EigenTensor<T, Rank>::From(*y, y_new_dims);
  auto out_t = EigenTensor<T, Rank>::From(*out, out_new_dims);

  Eigen::DSizes<int, Rank> x_bcast_dims;
  Eigen::DSizes<int, Rank> y_bcast_dims;
  Eigen::DSizes<int, Rank> out_bcast_dims;

  GetBraodcastDims<Rank>(x_new_dims, y_new_dims, &x_bcast_dims, &y_bcast_dims);
  std::vector<int64_t> new_dims_vec(Rank);
  for (int i = 0; i < Rank; ++i) {
    new_dims_vec[i] = std::max(x_new_dims[i], y_new_dims[i]);
    out_bcast_dims[i] = new_dims_vec[i];
  }
  framework::DDim new_dims = framework::make_ddim(new_dims_vec);

  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();
  x_grad->mutable_data<T>(context.GetPlace());
  y_grad->mutable_data<T>(context.GetPlace());
  auto x_grad_t = EigenTensor<T, Rank>::From(*x_grad, x_new_dims);  // change
  auto y_grad_t = EigenTensor<T, Rank>::From(*y_grad, y_new_dims);
  auto out_grad_t = EigenTensor<T, Rank>::From(*out_grad, out_new_dims);
  framework::Tensor grad;
  grad.mutable_data<T>(new_dims, context.GetPlace());
  auto grad_t = EigenTensor<T, Rank>::From(grad);  // change

  auto x_minux_y = x_t.broadcast(x_bcast_dims) - y_t.broadcast(y_bcast_dims);
  auto x_minux_y_abs = x_minux_y.abs();
  T eps = static_cast<T>(1.0e-10f);
  if (p == 0) {
    grad_t.device(place) = grad_t.setZero();
  } else if (p == INFINITY || p == -INFINITY) {
    // 1. sign(x-y)
    grad_t.device(place) = x_minux_y / (x_minux_y_abs + eps);
    // 2. z = |x-y|, z(z == out) = out_grad, z(z != out) = 0
    grad_t.device(place) =
        (x_minux_y_abs == out_t.broadcast(out_bcast_dims)).template cast<T>() *
        out_grad_t.broadcast(out_bcast_dims) * grad_t;
  } else {
    auto factor = (out_t.broadcast(out_bcast_dims) + eps).pow(1 - p);
    grad_t.device(place) = (x_minux_y.abs() + eps).pow(p - 2) * factor *
                           x_minux_y * out_grad_t.broadcast(out_bcast_dims);
  }

  Eigen::DSizes<int, Rank * 2> x_reshape_dims;
  Eigen::DSizes<int, Rank * 2> y_reshape_dims;
  Eigen::DSizes<int, Rank> reduce_dims;
  GetReshapeAndReduceDims<Rank>(x_new_dims, y_new_dims, x_bcast_dims,
                                y_bcast_dims, &x_reshape_dims, &y_reshape_dims,
                                &reduce_dims);

  x_grad_t.device(place) = grad_t.reshape(x_reshape_dims)
                               .sum(reduce_dims)
                               .reshape(x_grad_t.dimensions());
  y_grad_t.device(place) = -grad_t.reshape(y_reshape_dims)
                                .sum(reduce_dims)
                                .reshape(y_grad_t.dimensions());
}

template <typename DeviceContext, typename T>
class DistKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto x_rank = context.Input<Tensor>("X")->dims().size();
    auto y_rank = context.Input<Tensor>("Y")->dims().size();
    auto rank = std::max(x_rank, y_rank);
    PADDLE_ENFORCE_LE(rank, 6,
                      platform::errors::Unimplemented(
                          "Op(dist) only support tensors with no more than 6 "
                          "dimensions."));
    switch (rank) {
      case 1:
        DistFunction<DeviceContext, T, 1>(context);
        break;
      case 2:
        DistFunction<DeviceContext, T, 2>(context);
        break;
      case 3:
        DistFunction<DeviceContext, T, 3>(context);
        break;
      case 4:
        DistFunction<DeviceContext, T, 4>(context);
        break;
      case 5:
        DistFunction<DeviceContext, T, 5>(context);
        break;
      case 6:
        DistFunction<DeviceContext, T, 6>(context);
        break;
    }
  }
};

template <typename DeviceContext, typename T>
class DistGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto x_rank = context.Input<Tensor>("X")->dims().size();
    auto y_rank = context.Input<Tensor>("Y")->dims().size();
    auto rank = std::max(x_rank, y_rank);
    PADDLE_ENFORCE_LE(rank, 6,
                      platform::errors::Unimplemented(
                          "Op(dist) only support tensors with no more than 6 "
                          "dimensions."));
    switch (rank) {
      case 1:
        DistGradFunction<DeviceContext, T, 1>(context);
        break;
      case 2:
        DistGradFunction<DeviceContext, T, 2>(context);
        break;
      case 3:
        DistGradFunction<DeviceContext, T, 3>(context);
        break;
      case 4:
        DistGradFunction<DeviceContext, T, 4>(context);
        break;
      case 5:
        DistGradFunction<DeviceContext, T, 5>(context);
        break;
      case 6:
        DistGradFunction<DeviceContext, T, 6>(context);
        break;
    }
  }
};

}  // namespace operators
}  // namespace paddle
