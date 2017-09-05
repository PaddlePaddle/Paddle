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

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class MulKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* X = context.Input<Tensor>("X");
    const Tensor* Y = context.Input<Tensor>("Y");
    Tensor* Z = context.Output<Tensor>("Out");
    const Tensor X_matrix =
        X->dims().size() > 2
            ? framework::FlattenToMatrix<T>(
                  *X, context.template GetAttr<int>("x_num_row_dims"))
            : *X;
    const Tensor Y_matrix =
        Y->dims().size() > 2
            ? framework::FlattenToMatrix<T>(
                  *Y, context.template GetAttr<int>("y_num_row_dims"))
            : *Y;

    Z->mutable_data<T>(context.GetPlace());
    auto* device_context =
        const_cast<platform::DeviceContext*>(context.device_context_);
    math::matmul<Place, T>(X_matrix, false, Y_matrix, false, 1, Z, 0,
                           device_context);
  }
};

template <typename Place, typename T>
class MulGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int x_num_row_dims = ctx.template GetAttr<int>("x_num_row_dims");
    int y_num_row_dims = ctx.template GetAttr<int>("y_num_row_dims");
    const Tensor* X = ctx.Input<Tensor>("X");
    const Tensor* Y = ctx.Input<Tensor>("Y");
    const Tensor X_matrix =
        X->dims().size() > 2 ? framework::FlattenToMatrix<T>(*X, x_num_row_dims)
                             : *X;
    const Tensor Y_matrix =
        Y->dims().size() > 2 ? framework::FlattenToMatrix<T>(*Y, y_num_row_dims)
                             : *Y;
    const Tensor* dOut = ctx.Input<Tensor>(framework::GradVarName("Out"));

    Tensor* dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    Tensor* dY = ctx.Output<Tensor>(framework::GradVarName("Y"));
    dX->mutable_data<T>(ctx.GetPlace());
    dY->mutable_data<T>(ctx.GetPlace());
    Tensor dX_matrix = dX->dims().size() > 2
                           ? framework::FlattenToMatrix<T>(*dX, x_num_row_dims)
                           : *dX;
    Tensor dY_matrix = dY->dims().size() > 2
                           ? framework::FlattenToMatrix<T>(*dY, y_num_row_dims)
                           : *dY;
    auto* device_context =
        const_cast<platform::DeviceContext*>(ctx.device_context_);
    // dX = dOut * Y'. dX: M x K, dOut : M x N, Y : K x N
    math::matmul<Place, T>(*dOut, false, Y_matrix, true, 1, &dX_matrix, 0,
                           device_context);
    // dY = X' * dOut. dY: K x N, dOut : M x N, X : M x K
    math::matmul<Place, T>(X_matrix, true, *dOut, false, 1, &dY_matrix, 0,
                           device_context);
  }
};

}  // namespace operators
}  // namespace paddle
