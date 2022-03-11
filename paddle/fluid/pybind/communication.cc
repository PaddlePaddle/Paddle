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

void BindTCPStore(py::module *m) {
  auto Store =
      py::class_<distributed::Store, std::shared_ptr<distributed::Store>>(
          *m, "Store")
          .def(py::init<>())
          .def("set",
               [](distributed::Store &self, const std::string &key,
                  const std::string &value) {
                 std::vector<uint8_t> data(value.begin(), value.end());
                 self.set(key, data);
               },
               py::arg("key"), py::arg("value"),
               py::call_guard<py::gil_scoped_release>())
          .def("get",
               [](distributed::Store &self,
                  const std::string &key) -> py::bytes {
                 auto data = self.get(key);
                 return py::bytes(reinterpret_cast<char *>(data.data()),
                                  data.size());
               },
               py::arg("key"), py::call_guard<py::gil_scoped_release>())
          .def("add", &distributed::Store::add,
               py::call_guard<py::gil_scoped_release>())
          .def("wait", &distributed::Store::wait,
               py::call_guard<py::gil_scoped_release>());

  py::class_<TCPStore, std::shared_ptr<TCPStore>>(*m, "TCPStore", Store)
      .def(py::init([](std::string hostname, uint16_t port, bool is_master,
                       size_t world_size, std::chrono::seconds timeout) {
             return std::make_shared<TCPStore>(hostname, port, is_master,
                                               world_size, timeout);
           }),
           py::arg("hostname"), py::arg("port"), py::arg("is_master"),
           py::arg("world_size"),
           py::arg("timeout") = distributed::tcputils::kNoTimeout,
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace pybind
}  // namespace paddle
