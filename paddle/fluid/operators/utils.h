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

template <typename T>
inline std::vector<T> GetDataFromTensor(const framework::Tensor* x) {
  auto* data = x->data<T>();
  framework::Tensor cpu_attr_tensor;
  if (platform::is_gpu_place(x->place())) {
    TensorCopySync(*x, platform::CPUPlace(), &cpu_attr_tensor);
    data = cpu_attr_tensor.data<T>();
  }
  auto vec_data = std::vector<T>(data, data + x->numel());
  return vec_data;
}
template <typename T>
inline std::vector<T> GetDataFromTensorList(
    const std::vector<const framework::Tensor*>& list_tensor) {
  std::vector<T> vec_new_data;
  for (size_t i = 0; i < list_tensor.size(); ++i) {
    auto tensor = list_tensor[i];
    PADDLE_ENFORCE_EQ(
        tensor->dims(), framework::make_ddim({1}),
        "ShapeError: If the element type is Tensor, "
        "the element's shape must be [1]. But received the element's shape "
        "is [%s]",
        tensor->dims());
    if (platform::is_gpu_place(tensor->place())) {
      framework::Tensor temp;
      TensorCopySync(*tensor, platform::CPUPlace(), &temp);
      vec_new_data.push_back((*temp.data<T>()));
    } else {
      vec_new_data.push_back((*tensor->data<T>()));
    }
  }
  return vec_new_data;
}
}  // namespace operators
}  // namespace paddle
