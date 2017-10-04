/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

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

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class ConvShiftKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *X = context.Input<Tensor>("X");
    auto *Y = context.Input<Tensor>("Y");
    auto *Out = context.Output<framework::Tensor>("Out");
    Out->mutable_data<T>(context.GetPlace());

    auto x = EigenMatrix<T>::Reshape(*X, 1 /* num_col_dims */);
    auto y = EigenMatrix<T>::Reshape(*Y, 1 /* num_col_dims */);
    auto out = EigenMatrix<T>::Reshape(*Out, 1 /* num_col_dims */);

    auto place = context.GetEigenDevice<Place>();
    out.device(place) = out.constant(static_cast<T>(0));

    auto x_dims = X->dims();
    auto y_dims = Y->dims();
    size_t batch_size = x_dims[0];
    size_t x_width = x_dims[1];
    size_t y_width = y_dims[1];
    size_t y_half_width = (y_width - 1) / 2;
    for (size_t k = 0; k < batch_size; ++k) {
      for (size_t i = 0; i < x_width; ++i) {
        for (size_t j = 0; j < y_width; ++j) {
          int index = i + j - y_half_width;
          index = (index + x_width) % x_width;
          out(k, i) += x(k, index) * y(k, j);
        }
      }
    }
  }
};

template <typename Place, typename T>
class ConvShiftGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *X = context.Input<Tensor>("X");
    auto *Y = context.Input<Tensor>("Y");
    auto *dOut =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    framework::Tensor *dX =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    dX->mutable_data<T>(context.GetPlace());
    framework::Tensor *dY =
        context.Output<framework::Tensor>(framework::GradVarName("Y"));
    dY->mutable_data<T>(context.GetPlace());

    auto x = EigenMatrix<T>::Reshape(*X, 1 /* num_col_dims */);
    auto y = EigenMatrix<T>::Reshape(*Y, 1 /* num_col_dims */);
    auto dout = EigenMatrix<T>::Reshape(*dOut, 1 /* num_col_dims */);
    auto dx = EigenMatrix<T>::Reshape(*dX, 1 /* num_col_dims */);
    auto dy = EigenMatrix<T>::Reshape(*dY, 1 /* num_col_dims */);

    auto place = context.GetEigenDevice<Place>();
    dx.device(place) = dx.constant(static_cast<T>(0));
    dy.device(place) = dy.constant(static_cast<T>(0));

    auto x_dims = X->dims();
    auto y_dims = Y->dims();
    size_t batch_size = x_dims[0];
    size_t x_width = x_dims[1];
    size_t y_width = y_dims[1];
    size_t y_half_width = (y_width - 1) / 2;
    for (size_t k = 0; k < batch_size; ++k) {
      for (size_t i = 0; i < x_width; ++i) {
        for (size_t j = 0; j < y_width; ++j) {
          int index = i + j - y_half_width;
          index = (index + x_width) % x_width;
          dx(k, i) += dout(k, i) * y(k, j);
          dy(k, j) += x(k, index) * dout(k, i);
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
