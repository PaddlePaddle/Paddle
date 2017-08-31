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

template <typename Place, typename T>
class CosSimKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input_x = context.Input<Tensor>("X");
    auto* input_y = context.Input<Tensor>("Y");
    auto* output_z = context.Output<Tensor>("Out");
    output_z->mutable_data<T>(context.GetPlace());

    auto X = EigenTensor<T>::From(*input_x);
    auto Y = EigenTensor<T>::From(*input_y);
    auto Z = EigenTensor<T>::From(*input_z);

    Eigen::DSizes<int, 2> dims(X.dimensions(0), X.size() / X.dimensions(0));
    auto Xr = X.reshape(dims);
    auto Yr = Y.reshape(dims);

    auto XY_rowsum = (Xr * Yr).sum(Eigen::array<int, 1>({1}));
    auto XX_rowsum = (Xr * Xr).sum(Eigen::array<int, 1>({1}));
    auto YY_rowsum = (Yr * Yr).sum(Eigen::array<int, 1>({1}));

    auto place = context.GetEigenDevice<Place>();
    Z.device(place) = XY_rowsum / XX_rowsum.sqrt() / YY_rowsum.sqrt()
  }
};

template <typename Place, typename T>
class MulGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input_x = context.Input<Tensor>("X");
    auto* input_y = context.Input<Tensor>("Y");
    auto* output_z = context.Input<Tensor>("Out");
    auto* grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* grad_y = context.Output<Tensor>(framework::GradVarName("Y"));
    auto* grad_z = context.Input<Tensor>(framework::GradVarName("Out"));
    grad_x->mutable_data<T>(context.GetPlace());
    grad_y->mutable_data<T>(context.GetPlace());

    auto X = EigenTensor<T>::From(*input_x);
    auto Y = EigenTensor<T>::From(*input_y);
    auto Z = EigenTensor<T>::From(*output_z);
    auto dX = EigenTensor<T>::From(*grad_x);
    auto dY = EigenTensor<T>::From(*grad_y);
    auto dZ = EigenTensor<T>::From(*grad_z);

    Eigen::DSizes<int, 2> dims(X.dimensions(0), X.size() / X.dimensions(0));
    auto Xr = X.reshape(dims);
    auto Yr = Y.reshape(dims);
    auto dXr = dX.reshape(dims);
    auto dYr = dY.reshape(dims);

    auto XX_rowsum = (Xr * Xr).sum(Eigen::array<int, 1>({1}));
    auto YY_rowsum = (Yr * Yr).sum(Eigen::array<int, 1>({1}));

    Eigen::DSizes<int, 2> bcast(1, dims[1]);
    auto XY_norm_bcast = (XX_rowsum.sqrt() * YY_rowsum.sqrt()).broadcast(bcast);
    auto Z_bcast = Zr.broadcast(bcast) auto dZ_bcast = dZr.broadcast(bcast)

                                                           auto place =
        context.GetEigenDevice<Place>();
    dXr.device(place) = dZ_bcast * (Yr / XY_norm_bcast -
                                    Z_bcast * Xr / XX_row_sum.broadcast(bcast));
    dYr.device(place) = dZ_bcast * (Xr / XY_norm_bcast -
                                    Z_bcast * Yr / XX_row_sum.broadcast(bcast));
  }
};

}  // namespace operators
}  // namespace paddle
