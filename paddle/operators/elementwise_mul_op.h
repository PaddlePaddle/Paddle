/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include "paddle/operators/math/math_function.h"

#include <iostream>
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class ElementWiseMulKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* z = ctx.Output<Tensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());

    auto x_e = framework::EigenVector<T>::Flatten(*x);
    auto y_e = framework::EigenVector<T>::Flatten(*y);
    auto z_e = framework::EigenVector<T>::Flatten(*z);

    auto x_dims = x->dims();
    auto y_dims = y->dims();
    PADDLE_ENFORCE_GE(x_dims.size(), y_dims.size(),
                      "Rank of first input must >= rank of second input.")

    if (x_dims == y_dims || product(y_dims) == 1) {
      z_e.device(ctx.GetEigenDevice<Place>()) = x_e * y_e;
      return;
    }

    // TODO(gongweibao): if axis is optional?
    bool broadcast = ctx.template Attr<int>("broadcast");
    PADDLE_ENFORCE(broadcast, "Do you forget broadcast parameter?");

    int axis = ctx.template Attr<int>("axis");
    PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(),
                   "Axis should be in range [0, x_dims)");

    int pre = 1;
    int n = 1;
    int post = 1;
    for (int i = 0; i < axis; ++i) {
      pre *= x_dims[i];
    }
    for (int i = 0; i < y_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(x_dims[i + axis], y_dims[i],
                        "Broadcast dimension mismatch.");
      n *= y_dims[i];
    }
    for (int i = axis + y_dims.size(); i < x_dims.size(); ++i) {
      post *= x_dims[i];
    }

    Eigen::DSizes<int, 1> one_d(x_e.size());
    if (post == 1) {
      auto y_bcast = y_e.reshape(Eigen::DSizes<int, 2>(1, n))
                         .broadcast(Eigen::DSizes<int, 2>(pre, 1))
                         .reshape(one_d);
      z_e.device(ctx.GetEigenDevice<Place>()) = x_e * y_bcast;
      return;
    } else {
      auto y_bcast = y_e.reshape(Eigen::DSizes<int, 3>(1, n, 1))
                         .broadcast(Eigen::DSizes<int, 3>(pre, 1, post))
                         .reshape(one_d);
      z_e.device(ctx.GetEigenDevice<Place>()) = x_e * y_bcast;
      return;
    }
  }
};

template <typename Place, typename T>
class ElementWiseMulGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    dx->mutable_data<T>(ctx.GetPlace());
    dy->mutable_data<T>(ctx.GetPlace());

    auto x_e = framework::EigenVector<T>::Flatten(*x);
    auto y_e = framework::EigenVector<T>::Flatten(*y);
    auto dx_e = framework::EigenVector<T>::Flatten(*dx);
    auto dy_e = framework::EigenVector<T>::Flatten(*dy);
    auto dout_e = framework::EigenVector<T>::Flatten(*dout);

    dx_e.device(ctx.GetEigenDevice<Place>()) = dout_e * y_e;
    dy_e.device(ctx.GetEigenDevice<Place>()) = x_e * dout_e;
  }
};

}  // namespace operators
}  // namespace paddle
