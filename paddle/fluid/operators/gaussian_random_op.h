// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <random>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/random.h"
namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class GaussianRandomKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    float mean = context.Attr<float>("mean");
    float std = context.Attr<float>("std");
    auto* tensor = context.Output<framework::Tensor>("Out");
    T* data = tensor->mutable_data<T>(context.GetPlace());

    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));
    if (seed == 0) {
      seed = std::random_device()();
    }
    platform::RandomFill(context.template device_context<DeviceContext>(), seed,
                         platform::NormalDistribution<T>(mean, std), data,
                         tensor->numel());
  }
};

}  // namespace operators
}  // namespace paddle
