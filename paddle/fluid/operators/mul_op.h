/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

constexpr int kMULMKLDNNINT8 = 1;
constexpr int kMULMKLDNNFP32 = 2;

template <typename DeviceContext, typename T>
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

    auto blas = math::GetBlas<DeviceContext, T>(context);

    blas.MatMul(x_matrix, y_matrix, z);
    if (z_dim.size() != 2) {
      z->Resize(z_dim);
    }
  }
};

template <typename DeviceContext, typename T>
class MulGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int x_num_col_dims = ctx.template Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.template Attr<int>("y_num_col_dims");
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto x_matrix = x->dims().size() > 2
                        ? framework::ReshapeToMatrix(*x, x_num_col_dims)
                        : static_cast<const Tensor&>(*x);
    auto y_matrix = y->dims().size() > 2
                        ? framework::ReshapeToMatrix(*y, y_num_col_dims)
                        : static_cast<const Tensor&>(*y);
    auto* dout = ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));

    Tensor dout_mat;
    dout_mat.ShareDataWith(*dout);
    dout_mat.Resize({framework::flatten_to_2d(x->dims(), x_num_col_dims)[0],
                     framework::flatten_to_2d(y->dims(), y_num_col_dims)[1]});

    auto* dx = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::LoDTensor>(framework::GradVarName("Y"));

    if (dx != nullptr) {
      dx->set_lod(x->lod());
    }
    if (dy != nullptr) {
      dy->set_lod(y->lod());
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      Tensor dx_matrix = dx->dims().size() > 2
                             ? framework::ReshapeToMatrix(*dx, x_num_col_dims)
                             : *dx;

      // dx = dout * y'. dx: M x K, dout : M x N, y : K x N
      blas.MatMul(dout_mat, false, y_matrix, true, &dx_matrix);
    }
    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
      Tensor dy_matrix = dy->dims().size() > 2
                             ? framework::ReshapeToMatrix(*dy, y_num_col_dims)
                             : *dy;
      // dy = x' * dout. dy K x N, dout : M x N, x : M x K
      blas.MatMul(x_matrix, true, dout_mat, false, &dy_matrix);
    }
  }
};

template <typename DeviceContext, typename T>
class MulDoubleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int x_num_col_dims = ctx.template Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.template Attr<int>("y_num_col_dims");
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto x_mat = x->dims().size() > 2
                     ? framework::ReshapeToMatrix(*x, x_num_col_dims)
                     : static_cast<const Tensor&>(*x);
    auto y_mat = y->dims().size() > 2
                     ? framework::ReshapeToMatrix(*y, y_num_col_dims)
                     : static_cast<const Tensor&>(*y);

    const int m = framework::flatten_to_2d(x->dims(), x_num_col_dims)[0];
    const int n = framework::flatten_to_2d(y->dims(), y_num_col_dims)[1];

    auto* dout = ctx.Input<framework::LoDTensor>("DOut");
    Tensor dout_mat;
    dout_mat.ShareDataWith(*dout);
    dout_mat.Resize({m, n});

    auto* ddx = ctx.Input<framework::LoDTensor>("DDX");
    auto* ddy = ctx.Input<framework::LoDTensor>("DDY");

    auto* dx = ctx.Output<framework::LoDTensor>("DX");
    auto* dy = ctx.Output<framework::LoDTensor>("DY");
    auto* ddout = ctx.Output<framework::LoDTensor>("DDOut");

    Tensor ddout_mat;
    if (ddout) {
      ddout->set_lod(dout->lod());
      // allocate and reshape ddout
      ddout->mutable_data<T>(ctx.GetPlace());
      ddout_mat.ShareDataWith(*ddout);
      ddout_mat.Resize({m, n});
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    // a flag to specify whether ddout value has been set, if flag
    // is false, MatMul beta should be 0 to set ddout, if flag is
    // true, MatMul beta should be 1 to add result to ddout.
    bool ddout_flag = false;
    if (ddx) {
      auto ddx_mat = ddx->dims().size() > 2
                         ? framework::ReshapeToMatrix(*ddx, x_num_col_dims)
                         : static_cast<const Tensor&>(*ddx);

      // dy = ddx' * dout. dy : K x M, ddx' : K x M, dout : M x N
      if (dy) {
        dy->set_lod(y->lod());
        // allocate and reshape dy
        dy->mutable_data<T>(ctx.GetPlace());
        Tensor dy_mat = dy->dims().size() > 2
                            ? framework::ReshapeToMatrix(*dy, y_num_col_dims)
                            : *dy;
        blas.MatMul(ddx_mat, true, dout_mat, false, &dy_mat);
      }
      // ddout1 = ddx * y. ddx : M x K, y : K x N, ddout1 : M x N
      if (ddout) {
        blas.MatMul(ddx_mat, false, y_mat, false, static_cast<T>(1.0),
                    &ddout_mat, static_cast<T>(ddout_flag));
        ddout_flag = true;
      }
    }
    if (ddy) {
      auto ddy_mat = ddy->dims().size() > 2
                         ? framework::ReshapeToMatrix(*ddy, y_num_col_dims)
                         : static_cast<const Tensor&>(*ddy);
      // dx = dout * ddy'. dout : M x N, ddy' : N x K, dx : M x K
      if (dx) {
        dx->set_lod(x->lod());
        // allocate and reshape dx
        dx->mutable_data<T>(ctx.GetPlace());
        Tensor dx_mat = dx->dims().size() > 2
                            ? framework::ReshapeToMatrix(*dx, x_num_col_dims)
                            : *dx;
        blas.MatMul(dout_mat, false, ddy_mat, true, &dx_mat);
      }
      // ddout2 = x * ddy. x : M x K, ddy : K x N, ddout2 : M x N
      if (ddout) {
        blas.MatMul(x_mat, false, ddy_mat, false, static_cast<T>(1.0),
                    &ddout_mat, static_cast<T>(ddout_flag));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
