// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class EmptyKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& out = GET_DATA_SAFELY(context.Output<framework::Tensor>("Out"),
                                "Output", "Out", "empty");
    out.Resize(framework::make_ddim(context.Attr<std::vector<int>>("shape")));
    auto dtype = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("dtype"));

    // @NOTE
    // only use cpu device for uninitialized memory
    auto place = platform::CPUPlace();
    // auto place = context.GetPlace();
    out.mutable_data(place, dtype);
    // out.mutable_data(context.GetPlace(), dtype);
  }
};

}  // namespace operators
}  // namespace paddle
