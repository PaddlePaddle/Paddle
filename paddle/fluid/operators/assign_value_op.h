//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

template <typename T>
class AssignValueKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto shape = ctx.Attr<std::vector<int>>("shape");
    auto* out = ctx.Output<framework::Tensor>("Out");
    int dtype = ctx.Attr<int>("dtype");
    const char* value_name = nullptr;
    switch (dtype) {
      case framework::proto::VarType::INT32:
        value_name = "int32_values";
        break;
      case framework::proto::VarType::FP32:
        value_name = "fp32_values";
        break;
      default:
        PADDLE_THROW("Unsupported dtype for assign_value_op: %d", dtype);
        break;
    }
    auto values = ctx.Attr<std::vector<T>>(value_name);
    framework::TensorFromVector(values, ctx.device_context(), out);
    out->Resize(framework::make_ddim(shape));
  }
};

}  // namespace operators
}  // namespace paddle
