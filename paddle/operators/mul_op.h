/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   you may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANy KIND, either express or implied.
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
    const Tensor* x = context.Input<Tensor>("X");
    const Tensor* y = context.Input<Tensor>("Y");
    Tensor* Z = context.Output<Tensor>("Out");
    const Tensor x_matrix =
        x->dims().size() > 2
            ? framework::FlattenToMatrix<T>(
                  *x, context.template GetAttr<int>("x_num_row_dims"))
            : *x;
    const Tensor y_matrix =
        y->dims().size() > 2
            ? framework::FlattenToMatrix<T>(
                  *y, context.template GetAttr<int>("y_num_row_dims"))
            : *y;

    Z->mutable_data<T>(context.GetPlace());
    auto* device_context =
        const_cast<platform::DeviceContext*>(context.device_context_);
    math::matmul<Place, T>(x_matrix, false, y_matrix, false, 1, Z, 0,
                           device_context);
  }
};

template <typename Place, typename T>
class MulGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int x_num_row_dims = ctx.template GetAttr<int>("x_num_row_dims");
    int y_num_row_dims = ctx.template GetAttr<int>("y_num_row_dims");
    const Tensor* x = ctx.Input<Tensor>("X");
    const Tensor* y = ctx.Input<Tensor>("Y");
    const Tensor x_matrix =
        x->dims().size() > 2 ? framework::FlattenToMatrix<T>(*x, x_num_row_dims)
                             : *x;
    const Tensor y_matrix =
        y->dims().size() > 2 ? framework::FlattenToMatrix<T>(*y, y_num_row_dims)
                             : *y;
    const Tensor* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

    Tensor* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    Tensor* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    auto* device_context =
        const_cast<platform::DeviceContext*>(ctx.device_context_);
    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      Tensor dx_matrix = dx->dims().size() > 2 ? framework::FlattenToMatrix<T>(
                                                     *dx, x_num_row_dims)
                                               : *dx;
      // dx = dout * y'. dx: M x K, dout : M x N, y : K x N
      math::matmul<Place, T>(*dout, false, y_matrix, true, 1, &dx_matrix, 0,
                             device_context);
    }
    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
      Tensor dy_matrix = dy->dims().size() > 2 ? framework::FlattenToMatrix<T>(
                                                     *dy, y_num_row_dims)
                                               : *dy;
      // dy = x' * dout. dy K x N, dout : M x N, x : M x K
      math::matmul<Place, T>(x_matrix, true, *dout, false, 1, &dy_matrix, 0,
                             device_context);
    }
  }
};

}  // namespace operators
}  // namespace paddle
