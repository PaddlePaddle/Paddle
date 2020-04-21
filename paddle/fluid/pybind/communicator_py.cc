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

#include "paddle/fluid/pybind/communicator_py.h"

#include <Python.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/program_desc.h"
#include "pybind11/pybind11.h"

#include "paddle/fluid/operators/distributed/communicator.h"
#include "paddle/fluid/operators/distributed/communicator_common.h"

namespace py = pybind11;

using paddle::framework::ProgramDesc;
using paddle::framework::Scope;
using paddle::operators::distributed::AsyncCommunicator;
using paddle::operators::distributed::Communicator;
using paddle::operators::distributed::GeoSgdCommunicator;
using paddle::operators::distributed::HalfAsyncCommunicator;
using paddle::operators::distributed::SyncCommunicator;

using paddle::operators::distributed::CommContext;
using paddle::operators::distributed::RpcCtxMap;

namespace paddle {
namespace pybind {

void BindCommunicatorContext(py::module* m) {
  py::class_<CommContext>(*m, "CommContext")
      .def(
          py::init<const std::string&, const std::vector<std::string>&,
                   const std::vector<std::string>&, const std::vector<int64_t>&,
                   const std::vector<std::string>&, int, bool, bool>())
      .def("trainer_id",
           [](const CommContext& self) { return self.trainer_id; })
      .def("merged_var", [](const CommContext& self) { return self.var_name; })
      .def("__str__", [](const CommContext& self) { return self.str(); });
}

void BindCommunicator(py::module* m) {
  // Communicator is already used by nccl, change to DistCommunicator
  py::class_<Communicator, std::shared_ptr<Communicator>>(*m,
                                                          "DistCommunicator")
      .def(py::init([](const std::string& mode, const ProgramDesc& program,
                       Scope* param_scope,
                       std::map<std::string, std::string>& envs) {
        if (mode == "HALF_ASYNC") {
          Communicator::InitInstance<HalfAsyncCommunicator>(program,
                                                            param_scope, envs);
        } else if (mode == "ASYNC") {
          Communicator::InitInstance<AsyncCommunicator>(program, param_scope,
                                                        envs);
        } else if (mode == "GEO") {
          Communicator::InitInstance<GeoSgdCommunicator>(program, param_scope,
                                                         envs);
        } else if (mode == "SYNC") {
          Communicator::InitInstance<SyncCommunicator>(program, param_scope,
                                                       envs);
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "unsuported communicator MODE"));
        }
        return Communicator::GetInstantcePtr();
      }))
      .def(py::init([](const std::string& mode, const RpcCtxMap& send_ctx,
                       const RpcCtxMap& recv_ctx, Scope* param_scope,
                       std::map<std::string, std::string>& envs) {
        return Communicator::GetInstantcePtr();
      }))
      .def("stop", &Communicator::Stop)
      .def("start", &Communicator::Start)
      .def("is_running", &Communicator::IsRunning);
}
}  // namespace pybind
}  // namespace paddle
