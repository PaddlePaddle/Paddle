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
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FillConstantOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto data_type =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));
    auto value = ctx.Attr<float>("value");
    auto force_cpu = ctx.Attr<bool>("force_cpu");
    auto* out = ctx.Output<framework::Tensor>("Out");
    out->Resize(framework::make_ddim(ctx.Attr<std::vector<int>>("shape")));
    if (force_cpu) {
      auto cpu = platform::CPUPlace();
      out->mutable_data(cpu, framework::ToTypeIndex(data_type));
    } else {
      out->mutable_data(ctx.GetPlace(), framework::ToTypeIndex(data_type));
    }

    math::set_constant(ctx.template device_context<DeviceContext>(), out,
                       value);
  }
};

}  // namespace operators
}  // namespace paddle
