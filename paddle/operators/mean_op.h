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
#include "paddle/operators/type_alias.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class MeanKernel : public OpKernel {
public:
  void Compute(const ExecutionContext& context) const override {
    auto input = context.Input<Tensor>(0);
    auto output = context.Output<Tensor>(0);

    output->mutable_data<T>(context.GetPlace());

    auto X = EigenVector<T>::Flatten(*input);
    auto y = EigenScalar<T>::From(*output);
    auto place = context.GetEigenDevice<Place>();

    y.device(place) = X.mean();
  }
};

template <typename Place, typename T>
class MeanGradKernel : public OpKernel {
public:
  void Compute(const ExecutionContext& context) const override {
    auto OG = context.Input<Tensor>("Out" + framework::kGradVarSuffix);
    PADDLE_ENFORCE(framework::product(OG->dims()) == 1,
                   "Mean Gradient should be scalar");
    auto IG = context.Output<Tensor>("X" + framework::kGradVarSuffix);
    IG->mutable_data<T>(context.GetPlace());

    T ig_size = (T)framework::product(IG->dims());

    EigenVector<T>::Flatten(*IG).device(context.GetEigenDevice<Place>()) =
        EigenScalar<T>::From(*OG) / ig_size;
  }
};

}  // namespace operators
}  // namespace paddle
