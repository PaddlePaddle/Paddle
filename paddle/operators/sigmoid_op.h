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
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename Place, typename T>
class SigmoidKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto input = context.Input<Tensor>("X");
    auto output = context.Output<Tensor>("Y");
    output->mutable_data<T>(context.GetPlace());

    // The clipping is used in Paddle's raw implenmention
    auto X = EigenVector<T>::Flatten(*input);
    auto Y = EigenVector<T>::Flatten(*output);
    auto place = context.GetEigenDevice<Place>();

    Y.device(place) = 1. / (1. + (-X).exp());
  }
};

template <typename Place, typename T>
class SigmoidGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto Y_t = context.Input<Tensor>("Y");
    auto dY_t = context.Input<Tensor>(framework::GradVarName("Y"));
    auto dX_t = context.Output<Tensor>(framework::GradVarName("X"));

    dX_t->mutable_data<T>(context.GetPlace());

    auto dX = EigenVector<T>::Flatten(*dX_t);
    auto Y = EigenVector<T>::Flatten(*Y_t);
    auto dY = EigenVector<T>::Flatten(*dY_t);
    dX.device(context.GetEigenDevice<Place>()) = dY * Y * (1. - Y);
  }
};

}  // namespace operators
}  // namespace paddle
