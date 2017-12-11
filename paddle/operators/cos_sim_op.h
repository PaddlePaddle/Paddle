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
#include "paddle/operators/elementwise_add_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename T, typename DeviceContext>
void Function_forward(T* out, T* x_norm, T* y_norm,
                      ElementIterator<T, DeviceContext>& x,
                      ElementIterator<T, DeviceContext>& y, int row, int col) {
  for (int i = 0; i < row; ++i) {
    T xx = 0;
    T yy = 0;
    T xy = 0;
    for (int j = 0; j < col; ++j) {
      xy += (*x) * (*y);
      xx += (*x) * (*x);
      yy += (*y) * (*y);
      ++y;
      ++x;
    }
    x_norm[i] = sqrt(xx);
    y_norm[i] = sqrt(yy);

    out[i] = xy / (x_norm[i] * y_norm[i]);
  }
}

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
    auto x_iter = ElementIterator<T, DeviceContext>(in_x->data<T>(), rows_x,
                                                    cols, rows_x, cols);
    auto y_iter = ElementIterator<T, DeviceContext>(in_y->data<T>(), rows_y,
                                                    cols, rows_x, cols);

    Function_forward(out_z->data<T>(), out_x_norm->data<T>(),
                     out_y_norm->data<T>(), x_iter, y_iter, rows_x, cols);
    //
    //    // convert Tensor to Eigen Tensor
    ////    int rows_x = in_x->dims()[0];
    ////    int rows_y = in_y->dims()[0];
    //    auto x = EigenMatrix<T>::Reshape(*in_x, 1);
    //    auto y = EigenMatrix<T>::Reshape(*in_y, 1);
    //    auto z = EigenVector<T>::Flatten(*out_z);
    //    auto x_norm = EigenVector<T>::Flatten(*out_x_norm);
    //    auto y_norm = EigenVector<T>::Flatten(*out_y_norm);
    //
    //    // compute
    //    auto& place =
    //        *context.template device_context<DeviceContext>().eigen_device();
    //    auto row_along = Eigen::array<int, 1>({{1}});
    //    x_norm.device(place) = x.square().sum(row_along).sqrt();
    //    y_norm.device(place) = y.square().sum(row_along).sqrt();
    //    if (rows_x == rows_y) {
    //      auto xy = (x * y).sum(Eigen::array<int, 1>({{1}}));
    //      z.device(place) = xy / x_norm / y_norm;
    //    } else {
    //      Eigen::DSizes<int, 2> bcast(rows_x, 1);
    //      auto xy = (x * y.broadcast(bcast)).sum(row_along);
    //      z.device(place) = xy / x_norm / y_norm.broadcast(bcast);
    //    }
  }
};

template <typename T, typename DeviceContext>
void Function_element(T* result, ElementIterator<T, DeviceContext> dz,
                      ElementIterator<T, DeviceContext> y,
                      ElementIterator<T, DeviceContext> x_norm,
                      ElementIterator<T, DeviceContext> y_norm,
                      ElementIterator<T, DeviceContext> z,
                      ElementIterator<T, DeviceContext> x, int num, int block) {
  for (int i = 0; i < num; ++i) {
    result[i % block] += (*dz) * ((*y) / ((*x_norm) * (*y_norm)) -
                                  (*z) * (*x) / ((*x_norm) * (*x_norm)));
    ++dz;
    ++y;
    ++x_norm;
    ++y_norm;
    ++z;
    ++x;
  }
}

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

    //////////////////////////////
    // ##
    auto x_iter = ElementIterator<T, DeviceContext>(in_x->data<T>(), rows_x,
                                                    cols, rows_x, cols);
    auto y_iter = ElementIterator<T, DeviceContext>(in_y->data<T>(), rows_y,
                                                    cols, rows_x, cols);
    auto z_iter = ElementIterator<T, DeviceContext>(in_z->data<T>(), rows_x, 1,
                                                    rows_x, cols);
    auto dz_iter = ElementIterator<T, DeviceContext>(in_grad_z->data<T>(),
                                                     rows_x, 1, rows_x, cols);
    auto x_norm_iter = ElementIterator<T, DeviceContext>(
        in_x_norm->data<T>(), rows_x, 1, rows_x, cols);
    auto y_norm_iter = ElementIterator<T, DeviceContext>(
        in_y_norm->data<T>(), rows_y, 1, rows_x, cols);
    // ##
    //////////////////////////////
    // compute dx
    if (out_grad_x) {
      out_grad_x->mutable_data<T>(context.GetPlace());

      //////////////////////////////
      // ##
      Function_element(out_grad_x->data<T>(), dz_iter, y_iter, x_norm_iter,
                       y_norm_iter, z_iter, x_iter, rows_x * cols,
                       rows_x * cols);
      // ##
      //////////////////////////////
    }
    // compute dy
    if (out_grad_y) {
      out_grad_y->mutable_data<T>(context.GetPlace());

      //////////////////////////////
      // ##
      Function_element(out_grad_y->data<T>(), dz_iter, x_iter, y_norm_iter,
                       x_norm_iter, z_iter, y_iter, rows_x * cols,
                       rows_y * cols);
      // ##
      //////////////////////////////
    }
  }
};

}  // namespace operators
}  // namespace paddle
