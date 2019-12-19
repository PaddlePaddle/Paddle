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
#include <paddle/fluid/framework/operator.h>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

template <typename T = int32_t>
inline std::vector<T> GetDataFromTensor(const framework::Tensor* x) {
  std::vector<T> vec_new_data;
  if (x->type() == framework::proto::VarType::INT32) {
    auto* data = x->data<int>();
    framework::Tensor cpu_attr_tensor;
    if (platform::is_gpu_place(x->place())) {
      TensorCopySync(*x, platform::CPUPlace(), &cpu_attr_tensor);
      data = cpu_attr_tensor.data<int>();
    }
    vec_new_data = std::vector<T>(data, data + x->numel());
  } else if (x->type() == framework::proto::VarType::INT64) {
    auto* data = x->data<int64_t>();
    framework::Tensor cpu_attr_tensor;
    if (platform::is_gpu_place(x->place())) {
      TensorCopySync(*x, platform::CPUPlace(), &cpu_attr_tensor);
      data = cpu_attr_tensor.data<int64_t>();
    }
    vec_new_data = std::vector<T>(data, data + x->numel());
  } else {
    PADDLE_THROW("The dtype of Tensor must be int32 or int64.");
  }
  return vec_new_data;
}

template <typename T = int32_t>
inline std::vector<T> GetDataFromTensorList(
    const std::vector<const framework::Tensor*>& list_tensor) {
  std::vector<T> vec_new_data;
  for (size_t i = 0; i < list_tensor.size(); ++i) {
    auto tensor = list_tensor[i];
    PADDLE_ENFORCE_EQ(tensor->dims(), framework::make_ddim({1}),
                      "ShapeError: The shape of Tensor in list must be [1]. "
                      "But received the shape "
                      "is [%s]",
                      tensor->dims());

    if (tensor->type() == framework::proto::VarType::INT32) {
      if (platform::is_gpu_place(tensor->place())) {
        framework::Tensor temp;
        TensorCopySync(*tensor, platform::CPUPlace(), &temp);
        vec_new_data.push_back(static_cast<T>(*temp.data<int>()));
      } else {
        vec_new_data.push_back(static_cast<T>(*tensor->data<int>()));
      }
    } else if (tensor->type() == framework::proto::VarType::INT64) {
      if (platform::is_gpu_place(tensor->place())) {
        framework::Tensor temp;
        TensorCopySync(*tensor, platform::CPUPlace(), &temp);
        vec_new_data.push_back(static_cast<T>(*temp.data<int64_t>()));
      } else {
        vec_new_data.push_back(static_cast<T>(*tensor->data<int64_t>()));
      }
    } else {
      PADDLE_THROW("The dtype of Tensor in list must be int32 or int64.");
    }
  }
  return vec_new_data;
}
}  // namespace operators
}  // namespace paddle
