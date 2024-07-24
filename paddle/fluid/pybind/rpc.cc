//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/rpc.h"
#include "paddle/fluid/distributed/rpc/future_wrapper.h"
#include "paddle/fluid/distributed/rpc/python_rpc_handler.h"
#include "paddle/fluid/distributed/rpc/rpc_agent.h"

namespace py = pybind11;
using paddle::distributed::FutureWrapper;
using paddle::distributed::RpcAgent;
using paddle::distributed::WorkerInfo;
namespace paddle::pybind {

void BindWorkerInfo(py::module* m) {
  py::class_<WorkerInfo>(*m, "WorkerInfo")
      .def(py::init<std::string, uint32_t, std::string, uint32_t>())
      .def_readonly("name", &WorkerInfo::name_)
      .def_readonly("rank", &WorkerInfo::id_)
      .def_readonly("ip", &WorkerInfo::ip_)
      .def_readonly("port", &WorkerInfo::port_)
      .def("__str__", &WorkerInfo::to_string)
      .def("__repr__", &WorkerInfo::to_string);
}
void BindFuture(py::module* m) {
  py::class_<FutureWrapper, std::shared_ptr<FutureWrapper>>(*m, "Future")
      .def(py::init<>())
      .def("wait",
           &FutureWrapper::wait,
           py::call_guard<py::gil_scoped_release>());
}
void InitAndSetAgentInstance(py::module* m) {
  m->def(
      "init_and_set_agent_instance",
      [](const std::string& name, const std::vector<WorkerInfo>& infos) {
        auto instance = std::make_shared<RpcAgent>(name, infos);
        instance->SetAgentInstance(instance);
      },
      py::call_guard<py::gil_scoped_release>(),
      py::arg("name"),
      py::arg("infos"));
}
void InvokeRpc(py::module* m) {
  m->def(
      "invoke_rpc",
      [](const std::string& name, const std::string& py_func, int timeout_ms) {
        auto instance = RpcAgent::RpcAgentInstance();
        return std::make_shared<FutureWrapper>(
            instance->InvokeRpc(py_func, name, timeout_ms));
      },
      py::call_guard<py::gil_scoped_release>(),
      py::arg("to"),
      py::arg("py_func"),
      py::arg("timeout_ms"));
}
void StartWorker(py::module* m) {
  m->def(
      "rpc_start_worker",
      []() {
        auto instance = RpcAgent::RpcAgentInstance();
        instance->StartWorker();
      },
      py::call_guard<py::gil_scoped_release>());
}
void StartClient(py::module* m) {
  m->def(
      "rpc_start_client",
      []() {
        auto instance = RpcAgent::RpcAgentInstance();
        instance->StartClient();
      },
      py::call_guard<py::gil_scoped_release>());
}
void StopWorker(py::module* m) {
  m->def(
      "rpc_stop_worker",
      []() {
        auto instance = RpcAgent::RpcAgentInstance();
        instance->Stop();
      },
      py::call_guard<py::gil_scoped_release>());
}
void GetWorkerInfo(py::module* m) {
  m->def(
      "rpc_get_worker_info",
      [](const std::string& name) {
        auto instance = RpcAgent::RpcAgentInstance();
        return instance->GetWorkerInfo(name);
      },
      py::call_guard<py::gil_scoped_release>(),
      py::arg("name"));
}
void GetWorkerInfoByRank(py::module* m) {
  m->def(
      "rpc_get_worker_info_by_rank",
      [](uint32_t rank) {
        auto instance = RpcAgent::RpcAgentInstance();
        return instance->GetWorkerInfoById(rank);
      },
      py::call_guard<py::gil_scoped_release>(),
      py::arg("rank"));
}
void GetCurrentWorkerInfo(py::module* m) {
  m->def(
      "rpc_get_current_worker_info",
      []() {
        auto instance = RpcAgent::RpcAgentInstance();
        return instance->GetCurrentWorkerInfo();
      },
      py::call_guard<py::gil_scoped_release>());
}
void GetAllWorkerInfos(py::module* m) {
  m->def(
      "rpc_get_all_worker_infos",
      []() {
        auto instance = RpcAgent::RpcAgentInstance();
        return instance->GetAllWorkerInfos();
      },
      py::call_guard<py::gil_scoped_release>());
}
}  // namespace paddle::pybind
