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

#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {

inline framework::DDim GetShape(const framework::ExecutionContext &context,
                                std::string op_type) {
  // 1. shape is a Tensor
  if (context.HasInput("ShapeTensor")) {
    auto *shape_tensor = context.Input<framework::LoDTensor>("ShapeTensor");
    auto vec_shape = GetDataFromTensor<int>(shape_tensor);
    return framework::make_ddim(vec_shape);
  }

  // 2. shape is a list/tuple containing Tensor
  auto shape_tensor_list =
      context.MultiInput<framework::Tensor>("ShapeTensorList");
  if (shape_tensor_list.size() > 0) {
    auto vec_shape = GetDataFromTensorList(shape_tensor_list);
    return framework::make_ddim(vec_shape);
  }

  // 3. shape is a list/tuple without containing Tensor
  auto vec_shape = context.Attr<std::vector<int64_t>>("shape");
  return framework::make_ddim(vec_shape);
}

template <typename DeviceContext, typename T>
class EmptyKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto dtype = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("dtype"));

    framework::Tensor *out_tensor = nullptr;
    framework::Variable *out_var = context.OutputVar("Out");

    const std::string op_type = "empty";
    auto shape = GetShape(context, op_type);

    if (out_var->IsType<framework::LoDTensor>()) {
      out_tensor = out_var->GetMutable<framework::LoDTensor>();
      out_tensor->Resize(shape);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "In empty Op, the output only supports LoDTensor."));
    }

    // @NOTE
    // only use cpu device for uninitialized memory
    out_tensor->mutable_data(platform::CPUPlace(), dtype);
  }
};

}  // namespace operators
}  // namespace paddle
