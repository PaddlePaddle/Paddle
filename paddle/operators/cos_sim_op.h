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
#include "paddle/operators/elementwise_op_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename IT1, typename IT2, typename Callback>
static void ForEachZip(IT1 begin1, IT1 last1, IT2 begin2, Callback callback) {
  // This method could be implemented in CUDA
  for (; begin1 < last1; ++begin1, ++begin2) {
    callback(*begin1, *begin2);
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

  inline void operator()(T& x_norm, T& y_norm) const {
    size_t x_offset = &x_norm - x_norm_;
    size_t y_offset = &y_norm - y_norm_;

    auto* x = x_ + cols_ * x_offset;

    T xx = 0, xy = 0;
    T yy = 0;
    if (same_row) {
      auto* y = y_ + cols_ * y_offset;
      for (size_t i = 0; i < cols_; ++i) {
        xx += x[i] * x[i];
        yy += y[i] * y[i];
        xy += x[i] * y[i];
      }
      xx = sqrt(xx);
      yy = sqrt(yy);
      x_norm_[x_offset] = xx;
      y_norm_[y_offset] = yy;
      z_[x_offset] = xy / (xx * yy);
    } else {
      auto* y = y_;
      //      if (yy == -1) {
      //        yy = 0;
      //        for (size_t i = 0; i < cols_; ++i) {
      //          yy += y[i] * y[i];
      //        }
      //        y_norm[0] = sqrt(yy);
      //      }
      for (size_t i = 0; i < cols_; ++i) {
        xx += x[i] * x[i];
        yy += y[i] * y[i];  // only need
        xy += x[i] * y[i];
      }
      xx = sqrt(xx);
      yy = sqrt(yy);
      x_norm_[x_offset] = xx;
      y_norm_[0] = yy;
      z_[x_offset] = xy / (xx * yy);
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
      ForEachZip(out_x_norm->data<T>(), out_x_norm->data<T>() + rows_x,
                 out_y_norm->data<T>(), functor);
    } else {
      CosSimFunctor<T, false> functor(
          in_x->data<T>(), in_y->data<T>(), out_x_norm->data<T>(),
          out_y_norm->data<T>(), out_z->data<T>(), cols);
      ForEachZip(out_x_norm->data<T>(), out_x_norm->data<T>() + rows_x,
                 out_y_norm->data<T>(), functor);
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

  void operator()(const T& x_norm, const T& y_norm) const {
    size_t x_offset = &x_norm - x_norm_;
    size_t y_offset = &y_norm - y_norm_;

    auto x_norm_square = x_norm_[x_offset] * x_norm_[x_offset];
    //    auto y_norm_square = y_norm_[y_offset] * y_norm_[y_offset];
    auto xy_norm_prod = x_norm_[x_offset] * y_norm_[y_offset];
    auto dz = dz_[x_offset];

    auto* dx = dx_ + cols_ * x_offset;
    auto* x = x_ + cols_ * x_offset;
    auto* y = y_ + cols_ * y_offset;
    auto z = z_[x_offset];

    for (size_t i = 0; i < cols_; ++i) {
      dx[i] = dz * (y[i] / xy_norm_prod - z * x[i] / x_norm_square);
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

  void operator()(const T& x_norm, const T& y_norm) const {
    size_t x_offset = &x_norm - x_norm_;

    auto x_norm_square = x_norm_[x_offset] * x_norm_[x_offset];
    auto xy_norm_prod = x_norm_[x_offset] * y_norm_[0];
    auto dz = dz_[x_offset];
    auto z = z_[x_offset];

    auto* dx = dx_ + cols_ * x_offset;
    auto* x = x_ + cols_ * x_offset;

    for (size_t i = 0; i < cols_; ++i) {
      dx[i] = dz * (y_[i] / xy_norm_prod - z * x[i] / x_norm_square);
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
struct CosSimDyFunctor {
  CosSimDyFunctor(const T* x_norm, const T* y_norm, const T* x, const T* y,
                  const T* z, const T* dz, T* dy, int cols)
      : x_norm_(x_norm),
        y_norm_(y_norm),
        x_(x),
        y_(y),
        z_(z),
        dz_(dz),
        dy_(dy),
        cols_(static_cast<size_t>(cols)) {}

  void operator()(const T& x_norm, const T& y_norm) const {
    size_t x_offset = &x_norm - x_norm_;

    auto y_norm_square = y_norm_[0] * y_norm_[0];
    auto xy_norm_prod = x_norm_[x_offset] * y_norm_[0];
    auto dz = dz_[x_offset];
    auto z = z_[x_offset];
    auto* x = x_ + cols_ * x_offset;

    for (size_t i = 0; i < cols_; ++i) {
      dy_[i] += dz * (x[i] / xy_norm_prod - z * y_[i] / y_norm_square);
    }
  }

  const T* x_norm_;
  const T* y_norm_;
  const T* x_;
  const T* y_;
  const T* z_;
  const T* dz_;
  T* dy_;
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
        ForEachZip(in_x_norm->data<T>(), in_x_norm->data<T>() + rows_x,
                   in_y_norm->data<T>(), functor);
      }
      if (out_grad_y) {
        CosSimGradFunctor<T> functor(
            in_y_norm->data<T>(), in_x_norm->data<T>(), in_y->data<T>(),
            in_x->data<T>(), in_z->data<T>(), in_grad_z->data<T>(),
            out_grad_y->mutable_data<T>(context.GetPlace()), cols);
        ForEachZip(in_y_norm->data<T>(), in_y_norm->data<T>() + rows_x,
                   in_x_norm->data<T>(), functor);
      }
    } else {
      if (out_grad_x) {
        CosSimDxFunctor<T> functor(
            in_x_norm->data<T>(), in_y_norm->data<T>(), in_x->data<T>(),
            in_y->data<T>(), in_z->data<T>(), in_grad_z->data<T>(),
            out_grad_x->mutable_data<T>(context.GetPlace()), cols);
        ForEachZip(in_x_norm->data<T>(), in_x_norm->data<T>() + rows_x,
                   in_y_norm->data<T>(), functor);
      }
      if (out_grad_y) {
        CosSimDyFunctor<T> functor(
            in_x_norm->data<T>(), in_y_norm->data<T>(), in_x->data<T>(),
            in_y->data<T>(), in_z->data<T>(), in_grad_z->data<T>(),
            out_grad_y->mutable_data<T>(context.GetPlace()), cols);
        ForEachZip(in_x_norm->data<T>(), in_x_norm->data<T>() + rows_x,
                   in_y_norm->data<T>(), functor);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
