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
class PReluKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<Tensor>("X");
    auto* Out = context.Output<Tensor>("Out");

    Out->mutable_data<T>(context.GetPlace());

    auto alpha = static_cast<T>(context.Attr<float>("alpha"));

    auto X_vec = EigenVector<T>::Flatten(*X);
    auto Out_vec = EigenVector<T>::Flatten(*Out);

    // auto place = context.GetEigenDevice<Place>();
    // Out_vec.device(place)
    Out_vec = X_vec.cwiseMax(0.f) + X_vec.cwiseMin(0.f) * alpha;
  }
};

template <typename Place, typename T>
class PReluGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* dX = context.Output<Tensor>(framework::GradVarName("X"));
    auto* dO = context.Input<Tensor>(framework::GradVarName("Out"));

    auto* Out = context.Input<Tensor>("Out");

    auto alpha = static_cast<T>(context.Attr<float>("alpha"));

    dX->mutable_data<T>(context.GetPlace());
    for (int i = 0; i < dX->numel(); ++i) {
      if (Out->data<T>()[i] > 0) {
        dX->data<T>()[i] = dO->data<T>()[i];
      } else {
        dX->data<T>()[i] = dO->data<T>()[i] * alpha;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
