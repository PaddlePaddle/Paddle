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

paddle::Tensor custom_add(const paddle::Tensor& x, const paddle::Tensor& y) {
  return paddle::add(paddle::exp(x), paddle::exp(y));
}

PYBIND11_MODULE(custom_cpp_extension, m) {
  m.def("custom_add", &custom_add, "exp(x) + exp(y)");
  m.def("custom_sub", &custom_sub, "exp(x) - exp(y)");

  py::class_<Power>(m, "Power")
      .def(py::init<int, int>())
      .def(py::init<paddle::Tensor>())
      .def("forward", &Power::forward)
      .def("get", &Power::get);
}
