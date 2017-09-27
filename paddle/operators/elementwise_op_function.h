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
#include "paddle/framework/operator.h"

#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

/*
 * Out = X ⊙ Y
 * If Y's shape does not match X' shape, they will be reshaped.
 * For example:
 * 1. shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
 *    pre=2, n=3*4, post=5
 *    x.shape(2, 12, 5) * y.shape(1,12,1).broadcast(2,12,5)
 * 2. shape(X) = (2, 3, 4, 5), shape(Y) = (4,5)
 *    pre=2*3, n=4*5, post=1
 *    x.shape(2, 3, 20) * y.shape(1,1,20).broadcast(2,3,20)
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

#define EIGEN_FUNCTOR(name, eigen_op)                                          \
  struct Eigen##name##Functor {                                                \
    template <typename Place, typename T>                                      \
    inline void Run(const framework::Tensor* x, const framework::Tensor* y,    \
                    framework::Tensor* z,                                      \
                    const framework::ExecutionContext& ctx) {                  \
      auto x_e = framework::EigenVector<T>::Flatten(*x);                       \
      auto y_e = framework::EigenVector<T>::Flatten(*y);                       \
      auto z_e = framework::EigenVector<T>::Flatten(*z);                       \
      z_e.device(ctx.GetEigenDevice<Place>()) = eigen_op(x_e, y_e);            \
    }                                                                          \
    template <typename Place, typename T>                                      \
    inline void RunBroadCast(const framework::Tensor* x,                       \
                             const framework::Tensor* y, framework::Tensor* z, \
                             const framework::ExecutionContext& ctx, int pre,  \
                             int n) {                                          \
      auto x_e = framework::EigenVector<T>::Flatten(*x);                       \
      auto y_e = framework::EigenVector<T>::Flatten(*y);                       \
      auto z_e = framework::EigenVector<T>::Flatten(*z);                       \
      auto y_bcast = y_e.reshape(Eigen::DSizes<int, 2>(1, n))                  \
                         .broadcast(Eigen::DSizes<int, 2>(pre, 1))             \
                         .reshape(Eigen::DSizes<int, 1>(x_e.size()));          \
      z_e.device(ctx.GetEigenDevice<Place>()) = eigen_op(x_e, y_bcast);        \
    }                                                                          \
    template <typename Place, typename T>                                      \
    inline void RunBroadCast2(const framework::Tensor* x,                      \
                              const framework::Tensor* y,                      \
                              framework::Tensor* z,                            \
                              const framework::ExecutionContext& ctx, int pre, \
                              int n, int post) {                               \
      auto x_e = framework::EigenVector<T>::Flatten(*x);                       \
      auto y_e = framework::EigenVector<T>::Flatten(*y);                       \
      auto z_e = framework::EigenVector<T>::Flatten(*z);                       \
      auto y_bcast = y_e.reshape(Eigen::DSizes<int, 3>(1, n, 1))               \
                         .broadcast(Eigen::DSizes<int, 3>(pre, 1, post))       \
                         .reshape(Eigen::DSizes<int, 1>(x_e.size()));          \
      z_e.device(ctx.GetEigenDevice<Place>()) = eigen_op(x_e, y_bcast);        \
    }                                                                          \
  }

template <class functor, typename Place, typename T>
void ElementwiseCompute(const framework::ExecutionContext& ctx) {
  using Tensor = framework::Tensor;

  auto* x = ctx.Input<Tensor>("X");
  auto* y = ctx.Input<Tensor>("Y");
  auto* z = ctx.Output<Tensor>("Out");
  z->mutable_data<T>(ctx.GetPlace());

  auto x_dims = x->dims();
  auto y_dims = y->dims();
  PADDLE_ENFORCE_GE(x_dims.size(), y_dims.size(),
                    "Rank of first input must >= rank of second input.")

  if (x_dims == y_dims || product(y_dims) == 1) {
    functor f;
    f.template Run<Place, T>(x, y, z, ctx);
    return;
  }

  int axis = ctx.Attr<int>("axis");
  axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);
  PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(),
                 "Axis should be in range [0, x_dims)");

  int pre, n, post;
  get_mid_dims(x_dims, y_dims, axis, pre, n, post);
  if (post == 1) {
    functor f;
    f.template RunBroadCast<Place, T>(x, y, z, ctx, pre, n);
    return;
  } else {
    functor f;
    f.template RunBroadCast2<Place, T>(x, y, z, ctx, pre, n, post);
    return;
  }
}

#define EIGEN_ADD(x, y) ((x) + (y))
EIGEN_FUNCTOR(Add, EIGEN_ADD);

#define EIGEN_SUB(x, y) ((x) - (y))
EIGEN_FUNCTOR(Sub, EIGEN_SUB);

#define EIGEN_MUL(x, y) ((x) * (y))
EIGEN_FUNCTOR(Mul, EIGEN_MUL);

#define EIGEN_DIV(x, y) ((x) / (y))
EIGEN_FUNCTOR(Div, EIGEN_DIV);

template <typename Place, typename T, typename functor, typename functor1,
          typename broadcastfunctor, typename broadcast2functor>
void ElementwiseGradCompute(const framework::ExecutionContext& ctx) {
  using Tensor = framework::Tensor;

  auto* x = ctx.Input<Tensor>("X");
  auto* y = ctx.Input<Tensor>("Y");
  auto* out = ctx.Input<Tensor>("Out");
  auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

  auto place = ctx.GetEigenDevice<Place>();

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

  if (x_dims == y_dims) {
    functor f;
    f(place, x, y, out, dx, dy, dout);
    return;
  }

  if (product(y_dims) == 1) {
    functor1 f;
    f(place, x, y, out, dx, dy, dout);
    return;
  }

  int axis = ctx.Attr<int>("axis");
  axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);

  int pre, n, post;
  get_mid_dims(x_dims, y_dims, axis, pre, n, post);

  if (post == 1) {
    broadcastfunctor f;
    f(place, x, y, out, dx, dy, dout, pre, n);
    return;
  } else {
    broadcast2functor f;
    f(place, x, y, out, dx, dy, dout, pre, n, post);
    return;
  }
}
}  // namespace operators
}  // namespace paddle
