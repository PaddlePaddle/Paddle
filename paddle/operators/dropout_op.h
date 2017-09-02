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
class DropoutKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Output<Tensor>("Out");
    auto* mask = context.Output<Tensor>("Mask");
    mask->mutable_data<T>(context.GetPlace());
    y->mutable_data<T>(context.GetPlace());

    auto dims = x->dims();
    auto X = EigenMatrix<T>::From(*x);
    auto Y = EigenMatrix<T>::From(*y);
    auto M = EigenMatrix<T>::From(*mask);

    auto place = context.GetEigenDevice<Place>();
    M.device(place).setRandom<UniformRandomGenerator>();
    float dropout_prob = context.op_.GetAttr<float>("dropout_prob");
    M.device(place) = (M > dropout_prob).cast<float>();
    Y.device(place) = X * Y;
  }
};

template <typename Place, typename T>
class DropoutGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* grad_y = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* mask = context.Input<Tensor>("Mask");
    grad_x->mutable_data<T>(context.GetPlace());

    auto dims = grad_x->dims();
    auto M = EigenMatrix<T>::From(*mask);
    auto dX = EigenMatrix<T>::From(*grad_x);
    auto dY = EigenMatrix<T>::From(*grad_y);

    auto place = context.GetEigenDevice<Place>();
    dX.device(place) = dY * M;
  }
};

}  // namespace operators
}  // namespace paddle
