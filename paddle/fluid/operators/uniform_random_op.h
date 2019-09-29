// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

inline std::vector<int64_t> get_new_data_from_shape_tensor(
    const Tensor *new_data_tensor) {
  auto *new_data = new_data_tensor->data<int64_t>();
  if (platform::is_gpu_place(new_data_tensor->place())) {
    framework::Tensor cpu_starts_tensor;
    TensorCopySync(*new_data_tensor, platform::CPUPlace(), &cpu_starts_tensor);
    new_data = cpu_starts_tensor.data<int64_t>();
  }
  std::vector<int64_t> vec_new_data(new_data,
                                    new_data + new_data_tensor->numel());
  return vec_new_data;
}

inline std::vector<int64_t> get_new_shape_from_shape_tensorlist(
    const std::vector<const Tensor *> &list_new_shape_tensor) {
  std::vector<int64_t> vec_new_shape;
  vec_new_shape.reserve(list_new_shape_tensor.size());
  for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
    auto tensor = list_new_shape_tensor[i];
    PADDLE_ENFORCE_EQ(tensor->dims(), framework::make_ddim({1}),
                      "shape of dim tensor should be [1]");
    if (platform::is_gpu_place(tensor->place())) {
      framework::Tensor temp;
      TensorCopySync(*tensor, platform::CPUPlace(), &temp);

      vec_new_shape.push_back(static_cast<int64_t>(*temp.data<int64_t>()));
    } else {
      vec_new_shape.push_back(static_cast<int64_t>(*tensor->data<int64_t>()));
    }
  }

  return vec_new_shape;
}

}  // namespace operators
}  // namespace paddle
