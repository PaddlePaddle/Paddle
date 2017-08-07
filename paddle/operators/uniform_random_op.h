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
class UniformRandomKernel : public OpKernel {
 public:
  void Compute(const ExecutionContext &context) const override {
    auto tensor = context.Output<Tensor>(0);
    tensor->mutable_data<T>(context.GetPlace());

    auto eigenTensor = EigenVector<T>::Flatten(*tensor);
    auto dev = context.GetEigenDevice<Place>();
    auto min = context.op_.GetAttr<float>("min");
    auto max = context.op_.GetAttr<float>("max");
    auto seed = static_cast<uint64_t>(context.op_.GetAttr<int>("seed"));
    auto diff = max - min;
    Eigen::internal::UniformRandomGenerator<T> gen(seed);
    eigenTensor.device(dev) = eigenTensor.random(gen) * diff + min;
  }
};

}  // namespace operators
}  // namespace paddle
