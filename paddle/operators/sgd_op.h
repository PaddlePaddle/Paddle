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
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenScalar = framework::EigenScalar<T, MajorType, IndexType>;

template <typename Place, typename T>
class SGDOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto param = ctx.Input<Tensor>("Param");
    auto grad = ctx.Input<Tensor>("Grad");
    auto param_out = ctx.Output<Tensor>("ParamOut");
    auto learning_rate = ctx.Input<Tensor>("LearningRate");

    param_out->mutable_data<T>(ctx.GetPlace());

    auto p = EigenVector<T>::Flatten(*param);
    auto g = EigenVector<T>::Flatten(*grad);
    auto o = EigenVector<T>::Flatten(*param_out);
    auto lr = EigenScalar<T>::From(*learning_rate);
    auto place = ctx.GetEigenDevice<Place>();

    o.device(place) = p - lr * g;
  }
};

}  // namespace operators
}  // namespace paddle
