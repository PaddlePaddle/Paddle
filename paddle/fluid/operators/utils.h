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
inline T GetDataFromTensor(const framework::ExecutionContext& ctx,
                           std::string attr_name) {
  auto* attr_tensor = ctx.Input<framework::Tensor>(attr_name);
  PADDLE_ENFORCE_EQ(attr_tensor->dims(), framework::make_ddim({1}),
                    "ShapeError: The shape of %s must be [1]. "
                    "But received the shape is [%s].",
                    attr_name, attr_tensor->dims());
  auto* attr_data = attr_tensor->data<T>();
  framework::Tensor cpu_attr_tensor;
  if (platform::is_gpu_place(attr_tensor->place())) {
    TensorCopySync(*attr_tensor, platform::CPUPlace(), &cpu_attr_tensor);
    attr_data = cpu_attr_tensor.data<T>();
  }
  auto attr = std::vector<T>(attr_data, attr_data + attr_tensor->numel());
  return attr[0];
}
}  // namespace operators
}  // namespace paddle
