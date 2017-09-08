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
class CosSimKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input_x = context.Input<Tensor>("X");
    auto* input_y = context.Input<Tensor>("Y");
    auto* output_z = context.Output<Tensor>("Out");
    auto* output_x_norm = context.Output<Tensor>("XNorm");
    auto* output_y_norm = context.Output<Tensor>("YNorm");

    output_z->mutable_data<T>(context.GetPlace());
    output_x_norm->mutable_data<T>(context.GetPlace());
    output_y_norm->mutable_data<T>(context.GetPlace());

    auto dims = input_x->dims();
    int size = static_cast<int>(framework::product(dims));
    auto new_dims = framework::make_ddim({dims[0], size / dims[0]});
    auto x = EigenMatrix<T>::From(*input_x, new_dims);
    auto y = EigenMatrix<T>::From(*input_y, new_dims);
    auto z = EigenVector<T>::Flatten(*output_z);
    auto x_norm = EigenVector<T>::Flatten(*output_x_norm);
    auto y_norm = EigenVector<T>::Flatten(*output_y_norm);

    auto place = context.GetEigenDevice<Place>();
    auto xy = (x * y).sum(Eigen::array<int, 1>({{1}}));
    x_norm.device(place) = x.square().sum(Eigen::array<int, 1>({{1}})).sqrt();
    y_norm.device(place) = y.square().sum(Eigen::array<int, 1>({{1}})).sqrt();
    z.device(place) = xy / x_norm / y_norm;
  }
};

template <typename Place, typename T>
class CosSimGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input_x = context.Input<Tensor>("X");
    auto* input_y = context.Input<Tensor>("Y");
    auto* input_z = context.Input<Tensor>("Out");
    auto* input_x_norm = context.Input<Tensor>("XNorm");
    auto* input_y_norm = context.Input<Tensor>("YNorm");
    auto* output_grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* output_grad_y = context.Output<Tensor>(framework::GradVarName("Y"));
    auto* input_grad_z = context.Input<Tensor>(framework::GradVarName("Out"));

    auto dims = input_x->dims();
    int size = static_cast<int>(framework::product(dims));
    auto new_dims = framework::make_ddim({dims[0], size / dims[0]});
    auto x = EigenMatrix<T>::From(*input_x, new_dims);
    auto y = EigenMatrix<T>::From(*input_y, new_dims);
    auto z = EigenMatrix<T>::From(*input_z);
    auto x_norm = EigenMatrix<T>::From(*input_x_norm);
    auto y_norm = EigenMatrix<T>::From(*input_y_norm);
    auto dz = EigenMatrix<T>::From(*input_grad_z);

    Eigen::DSizes<int, 2> bcast(1, new_dims[1]);
    auto z_bcast = z.broadcast(bcast);
    auto dz_bcast = dz.broadcast(bcast);
    auto place = context.GetEigenDevice<Place>();
    auto x_snorm_bcast = x_norm.square().eval().broadcast(bcast);
    auto y_snorm_bcast = y_norm.square().eval().broadcast(bcast);
    auto norm_prod_bcast = (x_norm * y_norm).eval().broadcast(bcast);
    if (output_grad_x) {
      output_grad_x->mutable_data<T>(context.GetPlace());
      auto dx = EigenMatrix<T>::From(*output_grad_x, new_dims);
      dx.device(place) =
          dz_bcast * (y / norm_prod_bcast - z_bcast * x / x_snorm_bcast);
    }
    if (output_grad_y) {
      output_grad_y->mutable_data<T>(context.GetPlace());
      auto dy = EigenMatrix<T>::From(*output_grad_y, new_dims);
      dy.device(place) =
          dz_bcast * (x / norm_prod_bcast - z_bcast * y / y_snorm_bcast);
    }
  }
};

}  // namespace operators
}  // namespace paddle
