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
#include <vector>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/random.h"

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class UniformRandomKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    framework::Tensor* tensor = nullptr;
    auto out_var = ctx.OutputVar("Out");
    if (out_var->IsType<framework::LoDTensor>()) {
      tensor = out_var->GetMutable<framework::LoDTensor>();
    } else if (out_var->IsType<framework::SelectedRows>()) {
      auto shape = ctx.Attr<std::vector<int>>("shape");
      tensor = out_var->GetMutable<framework::SelectedRows>()->mutable_value();
      tensor->Resize(framework::make_ddim(shape));
    } else {
      PADDLE_THROW(
          "uniform_random_op's output only"
          "supports SelectedRows and Tensor");
    }
    unsigned int seed = static_cast<unsigned int>(ctx.Attr<int>("seed"));
    if (seed == 0) {
      seed = std::random_device()();
    }
    platform::UniformRealDistribution<T> dist(
        static_cast<T>(ctx.Attr<float>("min")),
        static_cast<T>(ctx.Attr<float>("max")));
    int64_t size = tensor->numel();
    T* data = tensor->mutable_data<T>(ctx.GetPlace());
    platform::RandomFill(ctx.template device_context<DeviceContext>(), seed,
                         dist, data, size);
  }
};

}  // namespace operators
}  // namespace paddle
