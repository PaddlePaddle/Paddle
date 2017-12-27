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
#include "paddle/framework/op_registry.h"
#include "paddle/operators/elementwise_op_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
struct CosSimDyFunctor {
  CosSimDyFunctor(const T* x_norm, const T* y_norm, const T* x, const T* y,
                  const T* z, const T* dz, T* dy, int cols);
  inline void operator()(size_t) const;
};

template <typename Callback>
static void ForEachZip(size_t num, Callback callback) {
  for (size_t i = 0; i < num; ++i) {
    callback(i);
  }
}

template <typename T, bool same_row>
struct CosSimFunctor {
  CosSimFunctor(const T* x, const T* y, T* x_norm, T* y_norm, T* z, int cols)
      : x_norm_(x_norm),
        y_norm_(y_norm),
        x_(x),
        y_(y),
        z_(z),
        cols_(static_cast<size_t>(cols)) {}

  inline HOSTDEVICE void operator()(size_t offset) const {
    auto* x = x_ + cols_ * offset;
    T xx = 0, xy = 0, yy = 0;
    if (same_row) {
      auto* y = y_ + cols_ * offset;
      for (size_t i = 0; i < cols_; ++i) {
        xx += x[i] * x[i];
        yy += y[i] * y[i];
        xy += x[i] * y[i];
      }
      xx = sqrt(xx);
      yy = sqrt(yy);
      y_norm_[offset] = yy;
      x_norm_[offset] = xx;
      z_[offset] = xy / (xx * yy);
    } else {  // This can be wrote in a better way.
      for (size_t i = 0; i < cols_; ++i) {
        xx += x[i] * x[i];
        yy += y_[i] * y_[i];  // only need
        xy += x[i] * y_[i];
      }
      xx = sqrt(xx);
      yy = sqrt(yy);
      y_norm_[0] = yy;
      x_norm_[offset] = xx;
      z_[offset] = xy / (xx * yy);
    }
  }

  T* x_norm_;
  T* y_norm_;
  const T* x_;
  const T* y_;
  T* z_;
  const size_t cols_;
};

template <typename DeviceContext, typename T>
class CosSimKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // get Tensor
    auto* in_x = context.Input<Tensor>("X");
    auto* in_y = context.Input<Tensor>("Y");
    auto* out_z = context.Output<Tensor>("Out");
    auto* out_x_norm = context.Output<Tensor>("XNorm");
    auto* out_y_norm = context.Output<Tensor>("YNorm");
    out_z->mutable_data<T>(context.GetPlace());
    out_x_norm->mutable_data<T>(context.GetPlace());
    out_y_norm->mutable_data<T>(context.GetPlace());

    int rows_x = in_x->dims()[0];
    int rows_y = in_y->dims()[0];

    int cols = framework::product(in_x->dims()) / rows_x;

    if (rows_x == rows_y) {
      CosSimFunctor<T, true> functor(
          in_x->data<T>(), in_y->data<T>(), out_x_norm->data<T>(),
          out_y_norm->data<T>(), out_z->data<T>(), cols);
      ForEachZip(rows_x, functor);
    } else {
      CosSimFunctor<T, false> functor(
          in_x->data<T>(), in_y->data<T>(), out_x_norm->data<T>(),
          out_y_norm->data<T>(), out_z->data<T>(), cols);
      ForEachZip(rows_x, functor);
    }
  }
};

template <typename T>
struct CosSimGradFunctor {
  CosSimGradFunctor(const T* x_norm, const T* y_norm, const T* x, const T* y,
                    const T* z, const T* dz, T* dx, int cols)
      : x_norm_(x_norm),
        y_norm_(y_norm),
        x_(x),
        y_(y),
        z_(z),
        dz_(dz),
        dx_(dx),
        cols_(static_cast<size_t>(cols)) {}

  inline HOSTDEVICE void operator()(size_t offset) const {
    auto x_norm_square = x_norm_[offset] * x_norm_[offset];
    auto xy_norm_prod = x_norm_[offset] * y_norm_[offset];
    auto dz = dz_[offset];
    auto z = z_[offset];

    auto* dx = dx_ + cols_ * offset;
    auto* x = x_ + cols_ * offset;
    auto* y = y_ + cols_ * offset;

    auto reciprocal_xy_norm_prod = 1 / xy_norm_prod;
    auto reciprocal_x_norm_square = 1 / x_norm_square;
    for (size_t i = 0; i < cols_; ++i) {
      dx[i] = dz * (y[i] * reciprocal_xy_norm_prod -
                    z * x[i] * reciprocal_x_norm_square);
    }
  }

  const T* x_norm_;
  const T* y_norm_;
  const T* x_;
  const T* y_;
  const T* z_;
  const T* dz_;
  T* dx_;
  const size_t cols_;
};

template <typename T>
struct CosSimDxFunctor {
  CosSimDxFunctor(const T* x_norm, const T* y_norm, const T* x, const T* y,
                  const T* z, const T* dz, T* dx, int cols)
      : x_norm_(x_norm),
        y_norm_(y_norm),
        x_(x),
        y_(y),
        z_(z),
        dz_(dz),
        dx_(dx),
        cols_(static_cast<size_t>(cols)) {}

  inline HOSTDEVICE void operator()(size_t offset) const {
    auto xy_norm_prod = x_norm_[offset] * y_norm_[0];
    auto dz = dz_[offset];
    auto z = z_[offset];
    auto* x = x_ + cols_ * offset;
    auto reciprocal_xy_norm_prod = 1 / xy_norm_prod;
    auto x_norm_square = x_norm_[offset] * x_norm_[offset];
    auto* dx = dx_ + cols_ * offset;
    auto reciprocal_x_norm_square = 1 / x_norm_square;

    for (size_t i = 0; i < cols_; ++i) {
      dx[i] = dz * (y_[i] * reciprocal_xy_norm_prod -
                    z * x[i] * reciprocal_x_norm_square);
    }
  }
  const T* x_norm_;
  const T* y_norm_;
  const T* x_;
  const T* y_;
  const T* z_;
  const T* dz_;
  T* dx_;
  const size_t cols_;
};

template <typename DeviceContext, typename T>
class CosSimGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // get Tensor
    auto* in_x = context.Input<Tensor>("X");
    auto* in_y = context.Input<Tensor>("Y");
    auto* in_z = context.Input<Tensor>("Out");
    auto* in_x_norm = context.Input<Tensor>("XNorm");
    auto* in_y_norm = context.Input<Tensor>("YNorm");
    auto* out_grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* out_grad_y = context.Output<Tensor>(framework::GradVarName("Y"));
    auto* in_grad_z = context.Input<Tensor>(framework::GradVarName("Out"));

    // compute gradident
    int rows_x = in_x->dims()[0];
    int rows_y = in_y->dims()[0];
    int cols = framework::product(in_x->dims()) / rows_x;

    if (rows_x == rows_y) {
      if (out_grad_x) {
        CosSimGradFunctor<T> functor(
            in_x_norm->data<T>(), in_y_norm->data<T>(), in_x->data<T>(),
            in_y->data<T>(), in_z->data<T>(), in_grad_z->data<T>(),
            out_grad_x->mutable_data<T>(context.GetPlace()), cols);
        ForEachZip(rows_x, functor);
      }
      if (out_grad_y) {
        CosSimGradFunctor<T> functor(
            in_y_norm->data<T>(), in_x_norm->data<T>(), in_y->data<T>(),
            in_x->data<T>(), in_z->data<T>(), in_grad_z->data<T>(),
            out_grad_y->mutable_data<T>(context.GetPlace()), cols);
        ForEachZip(rows_x, functor);
      }
    } else {
      if (out_grad_x) {
        CosSimDxFunctor<T> functor(
            in_x_norm->data<T>(), in_y_norm->data<T>(), in_x->data<T>(),
            in_y->data<T>(), in_z->data<T>(), in_grad_z->data<T>(),
            out_grad_x->mutable_data<T>(context.GetPlace()), cols);
        ForEachZip(rows_x, functor);
      }
      if (out_grad_y) {
        out_grad_y->mutable_data<T>(context.GetPlace());
        math::SetConstant<DeviceContext, T> set_zero;
        auto& dev_ctx = context.template device_context<DeviceContext>();
        set_zero(dev_ctx, out_grad_y, static_cast<T>(0));

        CosSimDyFunctor<DeviceContext, T> functor(
            in_x_norm->data<T>(), in_y_norm->data<T>(), in_x->data<T>(),
            in_y->data<T>(), in_z->data<T>(), in_grad_z->data<T>(),
            out_grad_y->data<T>(), cols);
        ForEachZip(rows_x, functor);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
