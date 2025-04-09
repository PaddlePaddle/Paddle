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

#include "paddle/fluid/pybind/communication.h"

#include <Python.h>
// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <chrono>
#include <memory>
#include <string>

#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/core/distributed/store/tcp_store.h"

namespace py = pybind11;

namespace paddle::pybind {

void BindCommContextManager(py::module *m) {
  auto P2POption = py::class_<phi::distributed::P2POption>(*m, "P2POption")
                       .def(py::init<>());

  auto CommContextManager =
      py::class_<phi::distributed::CommContextManager,
                 std::shared_ptr<phi::distributed::CommContextManager>>(
          *m, "CommContextManager")
          .def_static("set_device_id",
                      &phi::distributed::CommContextManager::SetDeviceId,
                      py::call_guard<py::gil_scoped_release>())
#if defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)
          .def_static(
              "create_nccl_comm_context",
              &phi::distributed::CommContextManager::CreateNCCLCommContext,
              py::arg("store"),
              py::arg("unique_comm_key"),
              py::arg("rank"),
              py::arg("size"),
              py::arg("hash_key") = "",
              py::arg("p2p_opt") = nullptr,
              py::arg("nccl_comm_init_option") = 0,
              py::call_guard<py::gil_scoped_release>())
#endif
#if defined(PADDLE_WITH_XPU_BKCL)
          .def_static(
              "create_bkcl_comm_context",
              &phi::distributed::CommContextManager::CreateBKCLCommContext,
              py::arg("store"),
              py::arg("unique_comm_key"),
              py::arg("rank"),
              py::arg("size"),
              py::arg("hash_key") = "",
              py::call_guard<py::gil_scoped_release>())
#endif
#if defined(PADDLE_WITH_GLOO)
          .def_static(
              "create_gloo_comm_context",
              &phi::distributed::CommContextManager::CreateGlooCommContext,
              py::call_guard<py::gil_scoped_release>())
#endif
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
          .def_static(
              "create_xccl_comm_context",
              &phi::distributed::CommContextManager::CreateXCCLCommContext,
              py::call_guard<py::gil_scoped_release>())
#endif
          .def("set_store", &phi::distributed::CommContextManager::SetStore);
}

using TCPStore = phi::distributed::TCPStore;

void BindTCPStore(py::module *m) {
  auto Store = py::class_<phi::distributed::Store,
                          std::shared_ptr<phi::distributed::Store>>(*m, "Store")
                   .def(py::init<>())
                   .def(
                       "set",
                       [](phi::distributed::Store &self,
                          const std::string &key,
                          const std::string &value) {
                         std::vector<uint8_t> data(value.begin(), value.end());
                         self.set(key, data);
                       },
                       py::arg("key"),
                       py::arg("value"),
                       py::call_guard<py::gil_scoped_release>())
                   .def(
                       "get",
                       [](phi::distributed::Store &self,
                          const std::string &key) -> py::bytes {
                         auto data = self.get(key);
                         std::string s(data.begin(), data.end());
                         py::gil_scoped_acquire acquire;
                         return py::bytes(s);
                       },
                       py::arg("key"),
                       py::call_guard<py::gil_scoped_release>())
                   .def("add",
                        &phi::distributed::Store::add,
                        py::call_guard<py::gil_scoped_release>())
                   .def("wait",
                        &phi::distributed::Store::wait,
                        py::call_guard<py::gil_scoped_release>());

  py::class_<TCPStore, std::shared_ptr<TCPStore>>(*m, "TCPStore", Store)
      .def(py::init([](std::string hostname,
                       uint16_t port,
                       bool is_master,
                       size_t world_size,
                       int timeout) {
             return std::make_shared<TCPStore>(
                 hostname, port, is_master, world_size, timeout);
           }),
           py::arg("hostname"),
           py::arg("port"),
           py::arg("is_master"),
           py::arg("world_size"),
           py::arg("timeout") = 900,
           py::call_guard<py::gil_scoped_release>());

  m->def("create_or_get_global_tcp_store",
         &phi::distributed::CreateOrGetGlobalTCPStore);
}

}  // namespace paddle::pybind
