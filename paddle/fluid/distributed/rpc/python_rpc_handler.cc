// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/rpc/python_rpc_handler.h"

namespace paddle::distributed {
constexpr auto kInternalModule = "paddle.distributed.rpc.internal";

py::object getFunction(const py::object& module, const char* name) {
  py::object fn = module.attr(name);
  return fn;
}

PythonRpcHandler::PythonRpcHandler() {
  py::gil_scoped_acquire ag;
  // import python module
  py::object rpc_internal = py::module::import(kInternalModule);
  py_run_function_ = getFunction(rpc_internal, "_run_py_func");
  py_serialize_ = getFunction(rpc_internal, "_serialize");
  py_deserialize_ = getFunction(rpc_internal, "_deserialize");
}

py::object PythonRpcHandler::RunPythonFunc(const py::object& python_func) {
  py::gil_scoped_acquire ag;
  return py_run_function_(python_func);
}

std::string PythonRpcHandler::Serialize(const py::object& obj) {
  py::gil_scoped_acquire ag;
  py::object res = py_serialize_(obj);
  return res.cast<std::string>();
}

py::object PythonRpcHandler::Deserialize(const std::string& obj) {
  py::gil_scoped_acquire ag;
  return py_deserialize_(py::bytes(obj));
}

std::shared_ptr<PythonRpcHandler> PythonRpcHandler::python_rpc_handler_ =
    nullptr;
std::mutex PythonRpcHandler::lock_;

std::shared_ptr<PythonRpcHandler> PythonRpcHandler::GetInstance() {
  if (python_rpc_handler_ == nullptr) {
    std::lock_guard<std::mutex> guard(lock_);
    if (python_rpc_handler_ == nullptr) {
      python_rpc_handler_ = std::make_shared<PythonRpcHandler>();
      return python_rpc_handler_;
    }
  }
  return python_rpc_handler_;
}

}  // namespace paddle::distributed
