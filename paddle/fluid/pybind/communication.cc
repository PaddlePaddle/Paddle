/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <Python.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <chrono>
#include <string>

#include "paddle/fluid/distributed/store/tcp_store.h"
#include "paddle/fluid/pybind/communication.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

using TCPStore = paddle::distributed::TCPStore;

void BindTCPStore(py::module* m) {
  py::class_<TCPStore>(*m, "TCPStore")
      .def(
          py::init<std::string, uint16_t, bool, size_t, std::chrono::seconds>())
      .def("add", &TCPStore::add)
      .def("get", &TCPStore::get);
}

}  // namespace pybind
}  // namespace paddle
