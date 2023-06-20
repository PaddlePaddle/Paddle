// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <vector>

#include "custom_power.h"  // NOLINT
#include "paddle/extension.h"

paddle::Tensor custom_sub(paddle::Tensor x, paddle::Tensor y);

paddle::Tensor relu_cuda_forward(const paddle::Tensor& x);

paddle::Tensor custom_add(const paddle::Tensor& x, const paddle::Tensor& y) {
  return x.exp() + y.exp();
}

std::vector<paddle::Tensor> custom_tensor(
    const std::vector<paddle::Tensor>& inputs) {
  std::vector<paddle::Tensor> out;
  out.reserve(inputs.size());
  for (const auto& input : inputs) {
    out.push_back(input + 1.0);
  }
  return out;
}

paddle::Tensor nullable_tensor(bool return_none = false) {
  paddle::Tensor t;
  if (!return_none) {
    t = paddle::ones({2, 2});
  }
  return t;
}

paddle::optional<paddle::Tensor> optional_tensor(bool return_option = false) {
  paddle::optional<paddle::Tensor> t;
  if (!return_option) {
    t = paddle::ones({2, 2});
  }
  return t;
}

PYBIND11_MODULE(custom_cpp_extension, m) {
  m.def("custom_add", &custom_add, "exp(x) + exp(y)");
  m.def("custom_sub", &custom_sub, "exp(x) - exp(y)");
  m.def("custom_tensor", &custom_tensor, "x + 1");
  m.def("nullable_tensor", &nullable_tensor, "returned Tensor might be None");
  m.def(
      "optional_tensor", &optional_tensor, "returned Tensor might be optional");
  m.def("relu_cuda_forward", &relu_cuda_forward, "relu(x)");

  py::class_<Power>(m, "Power")
      .def(py::init<int, int>())
      .def(py::init<paddle::Tensor>())
      .def("forward", &Power::forward)
      .def("get", &Power::get);
}
