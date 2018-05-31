/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T, typename AttrType = T>
class NormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const framework::Tensor* in_x = context.Input<framework::Tensor>("X");
    const framework::Tensor* scale = context.Input<framework::Tensor>("Scale");
    auto* out = context.Output<framework::Tensor>("Out");
    auto epsilon = static_cast<T>(context.Attr<AttrType>("epsilon"));
    out->mutable_data<T>(context.GetPlace());
    int batch_size = in_x->dims()[0];
    int channels = in_x->dims()[1];
    int height = in_x->dims()[2];
    int width = in_x->dims()[3];
    int fea_len = height * width;
    auto* place =
        context.template device_context<DeviceContext>().eigen_device();
    auto x =
        framework::EigenMatrix<T, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *in_x, framework::make_ddim({batch_size, fea_len * channels}));
    // get square
    framework::Tensor x_square;
    x_square.mutable_data<T>(in_x->dims(), context.GetPlace());
    auto x_square_eigen =
        framework::EigenMatrix<T, Eigen::RowMajor, Eigen::DenseIndex>::From(
            x_square, framework::make_ddim({batch_size, fea_len * channels}));
    x_square_eigen.device(*place) = x.square();
    auto scale_eigen =
        framework::EigenVector<T, Eigen::RowMajor, Eigen::DenseIndex>::Flatten(
            *scale);
    for (int n = 0; n < batch_size; ++n) {
      framework::Tensor in_x_batch = in_x->Slice(n, n + 1);
      auto in_x_batch_eigen =
          framework::EigenMatrix<T, Eigen::RowMajor, Eigen::DenseIndex>::From(
              in_x_batch, framework::make_ddim({channels, fea_len}));
      framework::Tensor x_square_batch = x_square.Slice(n, n + 1);
      auto x_square_batch_eigen =
          framework::EigenMatrix<T, Eigen::RowMajor, Eigen::DenseIndex>::From(
              x_square_batch, framework::make_ddim({channels, fea_len}));
      framework::Tensor out_batch = out->Slice(n, n + 1);
      auto out_batch_eigen =
          framework::EigenMatrix<T, Eigen::RowMajor, Eigen::DenseIndex>::From(
              out_batch, framework::make_ddim({channels, fea_len}));
      framework::Tensor tmp_tensor;
      tmp_tensor.mutable_data<T>(framework::make_ddim({1, fea_len}),
                                 context.GetPlace());
      auto tmp = framework::EigenVector<T, Eigen::RowMajor,
                                        Eigen::DenseIndex>::Flatten(tmp_tensor);
      // get colsum and sqrt , inverse
      auto dim = Eigen::array<int, 1>({{0}});
      tmp.device(*place) = x_square_batch_eigen.sum(dim);
      tmp.device(*place) = (tmp + epsilon).sqrt().inverse();
      Eigen::array<int, 2> broadcast_dim_col;
      broadcast_dim_col[1] = 1;
      broadcast_dim_col[0] = channels;
      out_batch_eigen.device(*place) =
          in_x_batch_eigen * (tmp.broadcast(broadcast_dim_col));
      Eigen::array<int, 2> broadcast_dim_row;
      broadcast_dim_row[1] = fea_len;
      broadcast_dim_row[0] = 1;
      out_batch_eigen.device(*place) =
          out_batch_eigen * (scale_eigen.broadcast(broadcast_dim_row));
    }
  }
};
template <typename DeviceContext, typename T, typename AttrType = T>
class NormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const framework::Tensor* in_x = context.Input<framework::Tensor>("X");
    const framework::Tensor* scale = context.Input<framework::Tensor>("Scale");
    const framework::Tensor* out_grad =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto epsilon = static_cast<T>(context.Attr<AttrType>("epsilon"));
    framework::Tensor* in_x_grad =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    in_x_grad->mutable_data<T>(context.GetPlace());
    int batch_size = in_x->dims()[0];
    int channels = in_x->dims()[1];
    int height = in_x->dims()[2];
    int width = in_x->dims()[3];
    int fea_len = height * width;
    auto* place =
        context.template device_context<DeviceContext>().eigen_device();

    auto scale_eigen =
        framework::EigenVector<T, Eigen::RowMajor, Eigen::DenseIndex>::Flatten(
            *scale);
    auto x =
        framework::EigenMatrix<T, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *in_x, framework::make_ddim({batch_size, fea_len * channels}));
    // get square
    framework::Tensor x_square;
    x_square.mutable_data<T>(in_x->dims(), context.GetPlace());
    auto x_square_eigen =
        framework::EigenMatrix<T, Eigen::RowMajor, Eigen::DenseIndex>::From(
            x_square, framework::make_ddim({batch_size, fea_len * channels}));
    x_square_eigen.device(*place) = x.square();

    for (int n = 0; n < batch_size; ++n) {
      framework::Tensor in_x_batch = in_x->Slice(n, n + 1);
      auto in_x_batch_eigen =
          framework::EigenMatrix<T, Eigen::RowMajor, Eigen::DenseIndex>::From(
              in_x_batch, framework::make_ddim({channels, fea_len}));
      framework::Tensor in_g_batch = in_x_grad->Slice(n, n + 1);
      auto in_g_batch_eigen =
          framework::EigenMatrix<T, Eigen::RowMajor, Eigen::DenseIndex>::From(
              in_g_batch, framework::make_ddim({channels, fea_len}));
      framework::Tensor x_square_batch = x_square.Slice(n, n + 1);
      auto x_square_batch_eigen =
          framework::EigenMatrix<T, Eigen::RowMajor, Eigen::DenseIndex>::From(
              x_square_batch, framework::make_ddim({channels, fea_len}));
      framework::Tensor outg_batch = out_grad->Slice(n, n + 1);
      auto outg_batch_eigen =
          framework::EigenMatrix<T, Eigen::RowMajor, Eigen::DenseIndex>::From(
              outg_batch, framework::make_ddim({channels, fea_len}));

      framework::Tensor tmp_tensor;
      tmp_tensor.mutable_data<T>(framework::make_ddim({1, fea_len}),
                                 context.GetPlace());
      auto tmp_eigen =
          framework::EigenVector<T, Eigen::RowMajor,
                                 Eigen::DenseIndex>::Flatten(tmp_tensor);
      auto dim = Eigen::array<int, 1>({{0}});
      tmp_eigen.device(*place) = (in_x_batch_eigen * outg_batch_eigen).sum(dim);
      framework::Tensor norm_tmp_tensor;
      norm_tmp_tensor.mutable_data<T>(framework::make_ddim({1, fea_len}),
                                      context.GetPlace());
      auto norm_tmp_eigen =
          framework::EigenVector<T, Eigen::RowMajor,
                                 Eigen::DenseIndex>::Flatten(norm_tmp_tensor);
      norm_tmp_eigen.device(*place) =
          (x_square_batch_eigen.sum(dim) + epsilon).sqrt();
      Eigen::array<int, 2> broadcast_dim_col;
      broadcast_dim_col[1] = 1;
      broadcast_dim_col[0] = channels;
      in_g_batch_eigen.device(*place) =
          in_x_batch_eigen * tmp_eigen.broadcast(broadcast_dim_col);
      in_g_batch_eigen.device(*place) =
          in_g_batch_eigen /
          (norm_tmp_eigen * norm_tmp_eigen).broadcast(broadcast_dim_col);
      in_g_batch_eigen.device(*place) = outg_batch_eigen - in_g_batch_eigen;
      // outg_batch_eigen + (in_g_batch_eigen * -1);
      in_g_batch_eigen.device(*place) =
          in_g_batch_eigen / norm_tmp_eigen.broadcast(broadcast_dim_col);
      Eigen::array<int, 2> broadcast_dim_row;
      broadcast_dim_row[1] = fea_len;
      broadcast_dim_row[0] = 1;
      in_g_batch_eigen.device(*place) =
          in_g_batch_eigen * (scale_eigen.broadcast(broadcast_dim_row));
    }
  }
};
}  // namespace operators
}  // namespace paddle
