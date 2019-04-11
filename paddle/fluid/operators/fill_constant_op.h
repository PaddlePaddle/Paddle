/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {
template <typename T>
class FillConstantKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto data_type =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));
    auto value = ctx.Attr<float>("value");
    auto force_cpu = ctx.Attr<bool>("force_cpu");

    framework::Tensor *tensor = nullptr;

    framework::Variable *out_var = ctx.OutputVar("Out");

    if (out_var->IsType<framework::LoDTensor>()) {
      tensor = out_var->GetMutable<framework::LoDTensor>();
      tensor->Resize(
          framework::make_ddim(ctx.Attr<std::vector<int64_t>>("shape")));
    } else if (out_var->IsType<framework::SelectedRows>()) {
      tensor = out_var->GetMutable<framework::SelectedRows>()->mutable_value();
      tensor->Resize(
          framework::make_ddim(ctx.Attr<std::vector<int64_t>>("shape")));
    } else {
      PADDLE_THROW(
          "fill constant op's output only"
          "supports SelectedRows and LoDTensor");
    }

    if (force_cpu) {
      tensor->mutable_data(platform::CPUPlace(), data_type);
    } else {
      tensor->mutable_data(ctx.GetPlace(), data_type);
    }

    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(ctx.GetPlace());
    math::set_constant(dev_ctx, tensor, value);
  }
};
}  // namespace operators
}  // namespace paddle
