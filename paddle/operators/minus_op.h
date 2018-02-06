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

template <typename Place, typename T>
class MinusKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* left_tensor = context.Input<framework::Tensor>("X");
    auto* right_tensor = context.Input<framework::Tensor>("Y");
    auto* out_tensor = context.Output<framework::Tensor>("Out");

    out_tensor->mutable_data<T>(context.GetPlace());
    auto& dev = context.GetEigenDevice<Place>();
    framework::EigenVector<T>::Flatten(*out_tensor).device(dev) =
        framework::EigenVector<T>::Flatten(*left_tensor) -
        framework::EigenVector<T>::Flatten(*right_tensor);
  }
};

}  // namespace operators
}  // namespace paddle
