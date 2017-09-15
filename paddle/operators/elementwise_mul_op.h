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
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {
/*
 * Out = X ⊙ Y
 * 1. shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
 *    pre=2, n=3*4, post=5
 * 2. shape(X) = (2, 3, 4, 5), shape(Y) = (4,5)
 *    pre=2*3, n=4*5, post=1
 */

inline void get_mid_dims(const framework::DDim& x_dims,
                         const framework::DDim& y_dims, const int axis,
                         int& pre, int& n, int& post) {
  pre = 1;
  n = 1;
  post = 1;
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
}

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

    int axis = ctx.Attr<int>("axis");
    axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);
    PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(),
                   "Axis should be in range [0, x_dims)");

    int pre, n, post;
    get_mid_dims(x_dims, y_dims, axis, pre, n, post);
    if (post == 1) {
      auto y_bcast = y_e.reshape(Eigen::DSizes<int, 2>(1, n))
                         .broadcast(Eigen::DSizes<int, 2>(pre, 1))
                         .reshape(Eigen::DSizes<int, 1>(x_e.size()));
      z_e.device(ctx.GetEigenDevice<Place>()) = x_e * y_bcast;
      return;
    } else {
      auto y_bcast = y_e.reshape(Eigen::DSizes<int, 3>(1, n, 1))
                         .broadcast(Eigen::DSizes<int, 3>(pre, 1, post))
                         .reshape(Eigen::DSizes<int, 1>(x_e.size()));
      z_e.device(ctx.GetEigenDevice<Place>()) = x_e * y_bcast;
      return;
    }
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
