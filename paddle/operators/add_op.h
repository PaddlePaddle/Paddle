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
class AddKernel : public OpKernel {
public:
  void Compute(const KernelContext& context) const override {
    auto input0 = context.Input(0)->Get<Tensor>();
    auto input1 = context.Input(1)->Get<Tensor>();
    auto output = context.Output(0)->GetMutable<Tensor>();

    output->mutable_data<T>(context.GetPlace());

    EigenVector<T>::Flatten(*output).device(
        *(context.GetEigenDevice<Place>())) =
        EigenVector<T>::Flatten(input0) + EigenVector<T>::Flatten(input1);
  }
};

}  // namespace operators
}  // namespace paddle
