// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class CPUSeedKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out = context.Output<Tensor>("Out");
    auto* out_data = out->mutable_data<T>(context.GetPlace());
    int user_seed = context.Attr<int>("seed");

    // NOTE: fixed seed should only be used in unittest or for debug.
    // Guarantee to use random seed in training.
    std::random_device rnd;
    int seed;
    if (user_seed != 0) {
      seed = user_seed;
    } else {
      seed = rnd();
    }
    out_data[0] = seed;
  }
};

}  // namespace operators
}  // namespace paddle
