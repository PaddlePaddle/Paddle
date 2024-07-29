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

#pragma once

#include <pybind11/pybind11.h>

#include <memory>
#include <mutex>
#include <string>

#include "paddle/common/macros.h"

namespace py = pybind11;

namespace paddle {
namespace distributed {

class PYBIND11_EXPORT PythonRpcHandler {
 public:
  PythonRpcHandler();
  ~PythonRpcHandler() = default;
  static std::shared_ptr<PythonRpcHandler> GetInstance();
  // Run a pickled Python function and return the result py::object
  py::object RunPythonFunc(const py::object& python_func);

  // Serialized a py::object into a string
  std::string Serialize(const py::object& obj);

  // Deserialize a string into a py::object
  py::object Deserialize(const std::string& obj);

 private:
  DISABLE_COPY_AND_ASSIGN(PythonRpcHandler);

  static std::shared_ptr<PythonRpcHandler> python_rpc_handler_;
  // Ref to `paddle.distributed.rpc.internal.run_py_func`.
  py::object py_run_function_;

  // Ref to `paddle.distributed.rpc.internal.serialize`.
  py::object py_serialize_;

  // Ref to `paddle.distributed.rpc.internal.deserialize`.
  py::object py_deserialize_;

  // Lock to protect initialization.
  static std::mutex lock_;
};

}  // namespace distributed
}  // namespace paddle
