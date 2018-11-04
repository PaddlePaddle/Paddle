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
        static_cast<framework::proto::VarType::Type>(Attr<int>("dtype"));
    auto value = Attr<float>("value");
    auto force_cpu = Attr<bool>("force_cpu");

    framework::Tensor *tensor = nullptr;

    auto &out_var = *scope.FindVar(Output("Out"));

    if (out_var.IsType<framework::LoDTensor>()) {
      tensor = out_var.GetMutable<framework::LoDTensor>();
      tensor->Resize(framework::make_ddim(Attr<std::vector<int64_t>>("shape")));
    } else if (out_var.IsType<framework::SelectedRows>()) {
      tensor = out_var.GetMutable<framework::SelectedRows>()->mutable_value();
      tensor->Resize(framework::make_ddim(Attr<std::vector<int64_t>>("shape")));
    } else {
      PADDLE_THROW(
          "fill constant op's output only"
          "supports SelectedRows and LoDTensor");
    }

    if (force_cpu) {
      auto cpu = platform::CPUPlace();
      tensor->mutable_data(cpu, framework::ToTypeIndex(data_type));
    } else {
      tensor->mutable_data(dev_place, framework::ToTypeIndex(data_type));
    }

    math::set_constant(ctx.template device_context<DeviceContext>(), out,
                       value);
  }
};

}  // namespace operators
}  // namespace paddle
