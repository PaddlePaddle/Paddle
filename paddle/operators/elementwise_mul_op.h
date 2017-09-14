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
#include <iostream>
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/elementwise_op.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {
template <typename Place, typename T>
class ElementWiseMulKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* z = ctx.Output<Tensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());

    auto x_e = framework::EigenVector<T>::Flatten(*x);
    int axis = ctx.Attr<int>("axis");
    auto z_e = framework::EigenVector<T>::Flatten(*z);
    z_e.device(ctx.GetEigenDevice<Place>()) = x_e * reshape(x, y, axis);
  }
};

template <typename Place, typename T>
class ElementWiseMulGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto x_e = framework::EigenVector<T>::Flatten(*x);
    auto y_e = framework::EigenVector<T>::Flatten(*y);
    auto dout_e = framework::EigenVector<T>::Flatten(*dout);

    auto x_dims = x->dims();
    auto y_dims = y->dims();

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
    }
    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
    }

    if (x_dims == y_dims || product(y_dims) == 1) {
      if (dx) {
        auto dx_e = framework::EigenVector<T>::Flatten(*dx);
        dx_e.device(ctx.GetEigenDevice<Place>()) = dout_e * y_e;
      }

      if (dy) {
        auto dy_e = framework::EigenVector<T>::Flatten(*dy);
        dy_e.device(ctx.GetEigenDevice<Place>()) = x_e * dout_e;
      }
      return;
    }

    int axis = ctx.Attr<int>("axis");
    axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);

    int pre, n, post;
    get_mid_dims(x_dims, y_dims, axis, pre, n, post);

    // TODO(gongweibao): wrap reshape to a function.
    if (post == 1) {
      auto y_e_bcast = y_e.reshape(Eigen::DSizes<int, 2>(1, n))
                           .broadcast(Eigen::DSizes<int, 2>(pre, 1))
                           .reshape(Eigen::DSizes<int, 1>(x_e.size()));
      if (dx) {
        auto dx_e = framework::EigenVector<T>::Flatten(*dx);
        dx_e.device(ctx.GetEigenDevice<Place>()) = dout_e * y_e_bcast;
      }

      if (dy) {
        auto dy_e = framework::EigenVector<T>::Flatten(*dy);
        dy_e.device(ctx.GetEigenDevice<Place>()) =
            (x_e * dout_e)
                .reshape(Eigen::DSizes<int, 2>(pre, n))
                .sum(Eigen::array<int, 1>{{0}});
      }
      return;
    } else {
      auto y_e_bcast = y_e.reshape(Eigen::DSizes<int, 3>(1, n, 1))
                           .broadcast(Eigen::DSizes<int, 3>(pre, 1, post))
                           .reshape(Eigen::DSizes<int, 1>(x_e.size()));
      if (dx) {
        auto dx_e = framework::EigenVector<T>::Flatten(*dx);
        dx_e.device(ctx.GetEigenDevice<Place>()) = dout_e * y_e_bcast;
      }

      if (dy) {
        auto dy_e = framework::EigenVector<T>::Flatten(*dy);
        dy_e.device(ctx.GetEigenDevice<Place>()) =
            (x_e * dout_e)
                .reshape(Eigen::DSizes<int, 3>(pre, n, post))
                .sum(Eigen::array<int, 2>{{0, 2}});
      }
      return;
    }
  }
};

}  // namespace operators
}  // namespace paddle
