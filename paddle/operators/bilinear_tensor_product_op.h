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

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename Place, typename T>
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
    auto place = ctx.GetEigenDevice<Place>();

    // Create the temporary variables.
    Tensor left_mul;
    left_mul.mutable_data<T>(framework::make_ddim({batch_size, weight_dims[2]}),
                             ctx.GetPlace());
    auto left_mul_mat = EigenMatrix<T>::From(left_mul);
    Tensor output_col;
    output_col.mutable_data<T>(framework::make_ddim({weight_dims[0]}),
                               ctx.GetPlace());
    auto output_col_vec = EigenVector<T>::From(output_col);

    for (size_t i = 0; i < weight_dims[0]; ++i) {
      Tensor weight_mat = weight->Slice(i, i + 1).Resize(
          framework::make_ddim({weight_dims[1], weight_dims[2]}));
      math::gemm<Place, T>(ctx.device_context(), CblasNoTrans, CblasNoTrans,
                           batch_size, weight_dims[2], weight_dims[1], 1,
                           x->data<T>(), weight_mat.data<T>(), 0,
                           left_mul.data<T>());
      output_col_vec = (left_mul_mat * y_mat).sum(Eigen::DSizes<int, 1>(1));
      for (size_t j = 0; j < batch_size; ++j) {
        output_mat(j, i) = output_col_vec(j);
      }
    }
    if (bias) {
      auto bias_vec = EigenMatrix<T>::From(*bias);
      Eigen::DSizes<int, 2> bcast(batch_size, 1);
      output_mat.device(place) = bias_vec.broadcast(bcast) + output_mat;
    } else {
      output_mat.device(place) = output_mat;
    }
  }
};

template <typename Place, typename T>
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

    auto x_mat = EigenMatrix<T>::From(*x);
    auto y_mat = EigenMatrix<T>::From(*y);
    auto d_out_mat = EigenMatrix<T>::From(*d_out);
    auto place = ctx.GetEigenDevice<Place>();

    // Create the temporary variables for gradient.
    Tensor x_scale;
    x_scale.mutable_data<T>(framework::make_ddim({batch_size, weight_dims[1]}),
                            ctx.GetPlace());
    auto x_scale_mat = EigenMatrix<T>::From(x_scale);
    Tensor y_scale;
    y_scale.mutable_data<T>(framework::make_ddim({batch_size, weight_dims[2]}),
                            ctx.GetPlace());
    auto y_scale_mat = EigenMatrix<T>::From(y_scale);

    math::SetConstant<Place, T> set_zero;

    // Set X@Grad be zero at first.
    if (d_x) {
      d_x->mutable_data<T>(ctx.GetPlace());
      set_zero(ctx.device_context(), d_x, static_cast<T>(0));
    }

    // Set Y@Grad be zero at first.
    if (d_y) {
      d_y->mutable_data<T>(ctx.GetPlace());
      set_zero(ctx.device_context(), d_y, static_cast<T>(0));
    }

    // Caculate the X@Grad and Y@Grad.
    if (d_x || d_y) {
      Eigen::DSizes<int, 2> bcast_for_x(1, weight_dims[2]);
      Eigen::DSizes<int, 2> bcast_for_y(1, weight_dims[1]);
      for (int i = 0; i < weight_dims[0]; ++i) {
        Tensor weight_i = weight->Slice(i, i + 1).Resize(
            framework::make_ddim({weight_dims[1], weight_dims[2]}));
        auto output_vec = d_out_mat.chip(i, 1);
        if (d_x) {
          y_scale_mat.device(place) =
              output_vec.reshape(Eigen::DSizes<int, 2>(batch_size, 1))
                  .broadcast(bcast_for_x) *
              y_mat;
          math::gemm<Place, T>(ctx.device_context(), CblasNoTrans, CblasTrans,
                               batch_size, weight_dims[1], weight_dims[2], 1,
                               y_scale.data<T>(), weight_i.data<T>(), 1,
                               d_x->data<T>());
        }
        if (d_y) {
          x_scale_mat.device(place) =
              output_vec.reshape(Eigen::DSizes<int, 2>(batch_size, 1))
                  .broadcast(bcast_for_y) *
              x_mat;
          math::gemm<Place, T>(ctx.device_context(), CblasNoTrans, CblasNoTrans,
                               batch_size, weight_dims[2], weight_dims[1], 1,
                               x_scale.data<T>(), weight_i.data<T>(), 1,
                               d_y->data<T>());
        }
      }
    }

    // Caculate the gradient of Weight.
    if (d_weight) {
      d_weight->mutable_data<T>(ctx.GetPlace());
      Eigen::DSizes<int, 2> bcast_for_weight(1, weight_dims[1]);
      for (int i = 0; i < weight_dims[0]; ++i) {
        Tensor d_weight_i = d_weight->Slice(i, i + 1).Resize(
            framework::make_ddim({weight_dims[1], weight_dims[2]}));
        auto output_vec = d_out_mat.chip(i, 1);
        x_scale_mat.device(place) =
            output_vec.reshape(Eigen::DSizes<int, 2>(batch_size, 1))
                .broadcast(bcast_for_weight) *
            x_mat;
        math::gemm<Place, T>(ctx.device_context(), CblasTrans, CblasNoTrans,
                             weight_dims[1], weight_dims[2], batch_size, 1,
                             x_scale.data<T>(), y->data<T>(), 0,
                             d_weight_i.data<T>());
      }
    }

    // Caculate the gradient of Bias.
    if (d_bias) {
      d_bias->mutable_data<T>(ctx.GetPlace());
      auto d_bias_mat = EigenMatrix<T>::From(*d_bias);
      d_bias_mat.device(place) = d_out_mat.sum(Eigen::DSizes<int, 1>(0));
    }
  }
};

}  // namespace operators
}  // namespace paddle
