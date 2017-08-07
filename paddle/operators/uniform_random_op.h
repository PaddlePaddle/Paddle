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
#include <random>
#include <type_traits>
#include "paddle/operators/type_alias.h"
namespace paddle {
namespace operators {

template <typename T>
class CPUUniformRandomKernel : public OpKernel {
 public:
  void Compute(const ExecutionContext& context) const override {
    auto* tensor = context.Output<Tensor>(0);
    T* data = tensor->mutable_data<T>(context.GetPlace());
    unsigned int seed =
        static_cast<unsigned int>(context.op_.GetAttr<int>("seed"));
    std::minstd_rand engine;
    if (seed == 0) {
      seed = std::random_device()();
    }
    engine.seed(seed);
    std::uniform_real_distribution<T> dist(static_cast<T>(context.op_.GetAttr<float>("min")),
                                           static_cast<T>(context.op_.GetAttr<float>("max")));
    for (ssize_t i = 0; i < framework::product(tensor->dims()); ++i) {
      data[i] = dist(engine);
    }
  }
};

}  // namespace operators
}  // namespace paddle
