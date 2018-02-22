/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   You may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include "paddle/operators/math/math_function.h"

#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename Place, typename T>
class MulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    const Tensor* y = context.Input<Tensor>("Y");
    Tensor* z = context.Output<Tensor>("Out");
    const Tensor x_matrix =
        x->dims().size() > 2
            ? framework::ReshapeToMatrix(
                  *x, context.template Attr<int>("x_num_col_dims"))
            : *x;
    const Tensor y_matrix =
        y->dims().size() > 2
            ? framework::ReshapeToMatrix(
                  *y, context.template Attr<int>("y_num_col_dims"))
            : *y;

    z->mutable_data<T>(context.GetPlace());
    auto z_dim = z->dims();
    if (z_dim.size() != 2) {
      z->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
    }
    math::matmul<Place, T>(context.device_context(), x_matrix, false, y_matrix,
                           false, 1, z, 0);
    if (z_dim.size() != 2) {
      z->Resize(z_dim);
    }
  }
};

template <typename Place, typename T>
class MulGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int x_num_col_dims = ctx.template Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.template Attr<int>("y_num_col_dims");
    const Tensor* x = ctx.Input<Tensor>("X");
    const Tensor* y = ctx.Input<Tensor>("Y");
    const Tensor x_matrix = x->dims().size() > 2
                                ? framework::ReshapeToMatrix(*x, x_num_col_dims)
                                : *x;
    const Tensor y_matrix = y->dims().size() > 2
                                ? framework::ReshapeToMatrix(*y, y_num_col_dims)
                                : *y;
    const Tensor* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

    Tensor dout_mat;
    dout_mat.ShareDataWith(*dout);
    dout_mat.Resize({framework::flatten_to_2d(x->dims(), x_num_col_dims)[0],
                     framework::flatten_to_2d(y->dims(), y_num_col_dims)[1]});

    Tensor* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    Tensor* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      Tensor dx_matrix = dx->dims().size() > 2
                             ? framework::ReshapeToMatrix(*dx, x_num_col_dims)
                             : *dx;

      // dx = dout * y'. dx: M x K, dout : M x N, y : K x N
      math::matmul<Place, T>(ctx.device_context(), dout_mat, false, y_matrix,
                             true, 1, &dx_matrix, 0);
    }
    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
      Tensor dy_matrix = dy->dims().size() > 2
                             ? framework::ReshapeToMatrix(*dy, y_num_col_dims)
                             : *dy;
      // dy = x' * dout. dy K x N, dout : M x N, x : M x K
      math::matmul<Place, T>(ctx.device_context(), x_matrix, true, dout_mat,
                             false, 1, &dy_matrix, 0);
    }
  }
};

}  // namespace operators
}  // namespace paddle
