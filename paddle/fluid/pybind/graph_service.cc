// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/graph_service.h"
#include "paddle/fluid/distributed/service/graph_service.cc"
namespace py = pybind11;
using paddle::distributed::graph_service;

namespace paddle {
namespace pybind {

void BindGraphService(py::module* m) {
  py::class_<graph_service>(*m, "graph_service")
      .def(py::init<>())
      .def("set_keys", &graph_service::set_keys)
      .def("get_keys", &graph_service::get_keys);
}

}  // namespace pybind
}