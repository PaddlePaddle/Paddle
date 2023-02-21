/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0(the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once
#include <paddle/fluid/framework/operator.h>

#include <string>
#include <vector>

namespace phi {
namespace funcs {

template <typename T = int32_t>
inline std::vector<T> GetDataFromDenseTensor(const phi::DenseTensor* x) {
  std::vector<T> vec_new_data;
  if (paddle::framework::TransToProtoVarType(x->dtype()) ==
      paddle::framework::proto::VarType::INT32) {
    auto* data = x->data<int>();
    phi::DenseTensor cpu_attr_tensor;
    if (!paddle::platform::is_cpu_place(x->place())) {
      paddle::framework::TensorCopySync(
          *x, paddle::platform::CPUPlace(), &cpu_attr_tensor);
      data = cpu_attr_tensor.data<int>();
    }
    vec_new_data = std::vector<T>(data, data + x->numel());
  } else if (paddle::framework::TransToProtoVarType(x->dtype()) ==
             paddle::framework::proto::VarType::INT64) {
    auto* data = x->data<int64_t>();
    phi::DenseTensor cpu_attr_tensor;
    if (!paddle::platform::is_cpu_place(x->place())) {
      paddle::framework::TensorCopySync(
          *x, paddle::platform::CPUPlace(), &cpu_attr_tensor);
      data = cpu_attr_tensor.data<int64_t>();
    }
    // NOTE: Converting int64 to int32 may cause data overflow.
    vec_new_data = std::vector<T>(data, data + x->numel());
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "The dtype of Tensor must be int32 or int64, but received: %s",
        paddle::framework::TransToProtoVarType(x->dtype())));
  }
  return vec_new_data;
}

}  // namespace funcs
}  // namespace phi
