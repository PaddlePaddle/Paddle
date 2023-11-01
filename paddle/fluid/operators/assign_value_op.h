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

#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

template <typename T>
typename std::enable_if<std::is_same<T, bool>::value>::type CopyVectorToTensor(
    const char* value_name,
    phi::DenseTensor* out,
    const framework::ExecutionContext& ctx) {
  // phi::DenseTensore dtype is vector<bool>, it will be converted to
  //  vector<int>.
  //  at the same time, we can not use vector<bool> to hold the value, because
  //  the c++ use bit value to replace byte value.
  auto values = ctx.Attr<std::vector<int>>(value_name);
  framework::TensorFromVector(values, ctx.device_context(), out);

  // use the array to replace to vector
  bool* array_ptr = new T[values.size()];
  for (unsigned int i = 0; i < values.size(); i++) {
    array_ptr[i] = static_cast<T>(values[i]);
  }
  framework::TensorFromArray(
      array_ptr, values.size(), ctx.device_context(), out);
  delete[] array_ptr;
}

template <typename T>
typename std::enable_if<!std::is_same<T, bool>::value>::type CopyVectorToTensor(
    const char* value_name,
    phi::DenseTensor* out,
    const framework::ExecutionContext& ctx) {
  auto values = ctx.Attr<std::vector<T>>(value_name);
  framework::TensorFromVector(values, ctx.device_context(), out);
}

template <typename T, typename Context>
typename std::enable_if<std::is_same<T, bool>::value>::type CopyVectorToTensor(
    const Context& dev_ctx,
    const std::vector<Scalar>& values,
    phi::DenseTensor* out) {
  // If attribute value dtype is vector<bool>, it will be converted to
  // vector<int>. at the same time, we can not use vector<bool> to hold
  // the value, because the c++ use bit value to replace byte value.
  std::vector<int> assign_values;
  assign_values.reserve(values.size());
  for (const auto& val : values) {
    assign_values.emplace_back(val.to<int>());
  }
  phi::TensorFromVector(assign_values, dev_ctx, out);

  // use the array to replace to vector
  bool* array_ptr = new T[assign_values.size()];
  for (unsigned int i = 0; i < assign_values.size(); i++) {
    array_ptr[i] = static_cast<T>(assign_values[i]);
  }
  phi::TensorFromArray(array_ptr, assign_values.size(), dev_ctx, out);
  delete[] array_ptr;
}

template <typename T, typename Context>
typename std::enable_if<!std::is_same<T, bool>::value>::type CopyVectorToTensor(
    const Context& dev_ctx,
    const std::vector<Scalar>& values,
    phi::DenseTensor* out) {
  std::vector<T> assign_values;
  assign_values.reserve(values.size());
  for (const auto& val : values) {
    assign_values.emplace_back(val.to<T>());
  }
  phi::TensorFromVector(assign_values, dev_ctx, out);
}

template <typename T>
class AssignValueKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto shape = ctx.Attr<std::vector<int>>("shape");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    int dtype = ctx.Attr<int>("dtype");
    const char* value_name = nullptr;
    switch (dtype) {
      case framework::proto::VarType::BOOL:
        value_name = "bool_values";
        break;
      case framework::proto::VarType::INT32:
        value_name = "int32_values";
        break;
      case framework::proto::VarType::FP32:
        value_name = "fp32_values";
        break;
      case framework::proto::VarType::FP64:
        value_name = "fp64_values";
        break;
      case framework::proto::VarType::INT64:
        value_name = "int64_values";
      case framework::proto::VarType::INT8:
        value_name = "int8_values";
        break;
      default:
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported data type(code %d) for AssignValue operator, only "
            "supports bool, int32, float32, float64, int8 and int64.",
            dtype));
        break;
    }
    CopyVectorToTensor<T>(value_name, out, ctx);
    out->Resize(phi::make_ddim(shape));
  }
};

}  // namespace operators
}  // namespace paddle
