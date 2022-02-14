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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
class BilinearTensorProductKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* weight = ctx.Input<Tensor>("Weight");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto y_mat = EigenMatrix<T>::From(*y);
    auto output_mat = EigenMatrix<T>::From(*out);

    auto batch_size = x->dims()[0];
    auto weight_dims = weight->dims();
    int out_dim = weight_dims[0];
    auto x_dim = weight_dims[1];
    auto y_dim = weight_dims[2];
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    // Create the intermediate variable to calculate the result of
    // Input(X) multiplied by Input(Weight_i), the formula is:
    // left_mul = X Weight_i.
    Tensor left_mul;
    left_mul.mutable_data<T>(framework::make_ddim({batch_size, y_dim}),
                             ctx.GetPlace());
    auto left_mul_mat = EigenMatrix<T>::From(left_mul);

    for (int i = 0; i < out_dim; ++i) {
      auto output_col_vec = output_mat.chip(i, 1);
      Tensor weight_mat =
          weight->Slice(i, i + 1).Resize(framework::make_ddim({x_dim, y_dim}));
      math::GetBlas<DeviceContext, T>(dev_ctx).GEMM(
          CblasNoTrans, CblasNoTrans, batch_size, y_dim, x_dim, 1, x->data<T>(),
          weight_mat.data<T>(), 0, left_mul.data<T>());
      output_col_vec.device(place) =
          (left_mul_mat * y_mat).sum(Eigen::DSizes<int, 1>(1));
    }
    if (bias) {
      auto bias_vec = EigenMatrix<T>::From(*bias);
      Eigen::DSizes<int, 2> bcast(batch_size, 1);
      output_mat.device(place) = bias_vec.broadcast(bcast) + output_mat;
    }
  }
};

template <typename DeviceContext, typename T>
class BilinearTensorProductGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* x = ctx.Input<Tensor>("X");
    const Tensor* y = ctx.Input<Tensor>("Y");
    const Tensor* weight = ctx.Input<Tensor>("Weight");
    Tensor* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    Tensor* d_y = ctx.Output<Tensor>(framework::GradVarName("Y"));
    Tensor* d_weight = ctx.Output<Tensor>(framework::GradVarName("Weight"));
    Tensor* d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));
    const Tensor* d_out = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto batch_size = x->dims()[0];
    auto weight_dims = weight->dims();
    int out_dim = weight_dims[0];
    auto x_dim = weight_dims[1];
    auto y_dim = weight_dims[2];

    auto x_mat = EigenMatrix<T>::From(*x);
    auto y_mat = EigenMatrix<T>::From(*y);
    auto d_out_mat = EigenMatrix<T>::From(*d_out);
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    // Create the intermediate variable to calculate the Output(Y@Grad).
    Tensor x_scale;
    x_scale.mutable_data<T>(framework::make_ddim({batch_size, x_dim}),
                            ctx.GetPlace());
    auto x_scale_mat = EigenMatrix<T>::From(x_scale);

    // Create the intermediate variable to calculate the Output(X@Grad).
    Tensor y_scale;
    y_scale.mutable_data<T>(framework::make_ddim({batch_size, y_dim}),
                            ctx.GetPlace());
    auto y_scale_mat = EigenMatrix<T>::From(y_scale);

    pten::funcs::SetConstant<DeviceContext, T> set_zero;

    if (d_x) {
      d_x->mutable_data<T>(ctx.GetPlace());
      set_zero(dev_ctx, d_x, static_cast<T>(0));
    }

    if (d_y) {
      d_y->mutable_data<T>(ctx.GetPlace());
      set_zero(dev_ctx, d_y, static_cast<T>(0));
    }

    if (d_weight) {
      d_weight->mutable_data<T>(ctx.GetPlace());
    }

    auto blas = math::GetBlas<DeviceContext, T>(ctx);

    // Caculate the Output(X@Grad) and Output(Y@Grad).
    if (d_x || d_y || d_weight) {
      Eigen::DSizes<int, 2> bcast_for_x(1, y_dim);
      Eigen::DSizes<int, 2> bcast_for_y(1, x_dim);
      Eigen::DSizes<int, 2> bcast_for_weight(1, x_dim);

      for (int i = 0; i < out_dim; ++i) {
        Tensor weight_i = weight->Slice(i, i + 1).Resize(
            framework::make_ddim({x_dim, y_dim}));
        auto output_vec = d_out_mat.chip(i, 1);

        if (d_x) {
          y_scale_mat.device(place) =
              output_vec.reshape(Eigen::DSizes<int, 2>(batch_size, 1))
                  .broadcast(bcast_for_x) *
              y_mat;
          blas.GEMM(CblasNoTrans, CblasTrans, batch_size, x_dim, y_dim, 1,
                    y_scale.data<T>(), weight_i.data<T>(), 1, d_x->data<T>());
        }

        if (d_y || d_weight) {
          auto output_vec_y =
              output_vec.reshape(Eigen::DSizes<int, 2>(batch_size, 1))
                  .broadcast(bcast_for_y);
          x_scale_mat.device(place) = output_vec_y * x_mat;
          if (d_y) {
            blas.GEMM(CblasNoTrans, CblasNoTrans, batch_size, y_dim, x_dim, 1,
                      x_scale.data<T>(), weight_i.data<T>(), 1, d_y->data<T>());
          }
          if (d_weight) {
            Tensor d_weight_i = d_weight->Slice(i, i + 1).Resize(
                framework::make_ddim({x_dim, y_dim}));
            blas.GEMM(CblasTrans, CblasNoTrans, x_dim, y_dim, batch_size, 1,
                      x_scale.data<T>(), y->data<T>(), 0, d_weight_i.data<T>());
          }
        }
      }
    }

    // calculate the gradient of Input(Bias).
    if (d_bias) {
      d_bias->mutable_data<T>(ctx.GetPlace());
      auto d_bias_mat = framework::EigenVector<T>::Flatten(*d_bias);
      d_bias_mat.device(place) = d_out_mat.sum(Eigen::DSizes<int, 1>(0));
    }
  }
};

}  // namespace operators
}  // namespace paddle
