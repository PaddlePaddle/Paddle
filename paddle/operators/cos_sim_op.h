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

    // convert Tensor to Eigen Tensor
    int rows_x = in_x->dims()[0];
    int rows_y = in_y->dims()[0];
    auto x = EigenMatrix<T>::Reshape(*in_x, 1);
    auto y = EigenMatrix<T>::Reshape(*in_y, 1);
    auto z = EigenVector<T>::Flatten(*out_z);
    auto x_norm = EigenVector<T>::Flatten(*out_x_norm);
    auto y_norm = EigenVector<T>::Flatten(*out_y_norm);

    // compute
    auto place = context.GetEigenDevice<Place>();
    auto row_along = Eigen::array<int, 1>({{1}});
    x_norm.device(place) = x.square().sum(row_along).sqrt();
    y_norm.device(place) = y.square().sum(row_along).sqrt();
    if (rows_x == rows_y) {
      auto xy = (x * y).sum(Eigen::array<int, 1>({{1}}));
      z.device(place) = xy / x_norm / y_norm;
    } else {
      Eigen::DSizes<int, 2> bcast(rows_x, 1);
      auto xy = (x * y.broadcast(bcast)).sum(row_along);
      z.device(place) = xy / x_norm / y_norm.broadcast(bcast);
    }
  }
};

template <typename Place, typename T>
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

    // convert Tensor to Eigen Tensor
    auto x = EigenMatrix<T>::Reshape(*in_x, 1);
    auto y = EigenMatrix<T>::Reshape(*in_y, 1);
    auto z = EigenMatrix<T>::Reshape(*in_z, 1);
    auto x_norm = EigenMatrix<T>::Reshape(*in_x_norm, 1);
    auto y_norm = EigenMatrix<T>::Reshape(*in_y_norm, 1);
    auto dz = EigenMatrix<T>::Reshape(*in_grad_z, 1);

    // compute gradident
    int rows_x = in_x->dims()[0];
    int rows_y = in_y->dims()[0];
    int cols = framework::product(in_x->dims()) / rows_x;
    Eigen::DSizes<int, 2> bcast_cols(1, cols);
    auto z_bcast = z.broadcast(bcast_cols);
    auto dz_bcast = dz.broadcast(bcast_cols);
    auto x_snorm_bcast = x_norm.square().eval().broadcast(bcast_cols);
    auto place = context.GetEigenDevice<Place>();
    if (rows_x == rows_y) {
      auto y_snorm_bcast = y_norm.square().eval().broadcast(bcast_cols);
      auto norm_prod_bcast = (x_norm * y_norm).eval().broadcast(bcast_cols);
      // compute dx
      if (out_grad_x) {
        out_grad_x->mutable_data<T>(context.GetPlace());
        auto dx = EigenMatrix<T>::Reshape(*out_grad_x, 1);
        auto grad = y / norm_prod_bcast - z_bcast * x / x_snorm_bcast;
        dx.device(place) = dz_bcast * grad;
      }
      // compute dy
      if (out_grad_y) {
        out_grad_y->mutable_data<T>(context.GetPlace());
        auto dy = EigenMatrix<T>::Reshape(*out_grad_y, 1);
        auto grad = x / norm_prod_bcast - z_bcast * y / y_snorm_bcast;
        dy.device(place) = dz_bcast * grad;
      }
    } else {
      Eigen::DSizes<int, 2> bcast_rows(rows_x, 1);
      Eigen::DSizes<int, 2> bcast_rows_cols(rows_x, cols);
      auto y_bcast = y.broadcast(bcast_rows);
      auto y_snorm_bcast = y_norm.square().eval().broadcast(bcast_rows_cols);
      auto norm_prod_bcast = (x_norm * y_norm.eval().broadcast(bcast_rows))
                                 .eval()
                                 .broadcast(bcast_cols);
      // compute dx
      if (out_grad_x) {
        out_grad_x->mutable_data<T>(context.GetPlace());
        auto dx = EigenMatrix<T>::Reshape(*out_grad_x, 1);
        auto grad = y_bcast / norm_prod_bcast - z_bcast * x / x_snorm_bcast;
        dx.device(place) = dz_bcast * grad;
      }
      // compute dy
      if (out_grad_y) {
        out_grad_y->mutable_data<T>(context.GetPlace());
        auto dy = EigenMatrix<T>::Reshape(*out_grad_y, 1);
        auto grad = x / norm_prod_bcast - z_bcast * y_bcast / y_snorm_bcast;
        dy.device(place) = (dz_bcast * grad).sum(Eigen::array<int, 1>({{0}}));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
