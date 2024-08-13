// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <unordered_map>

#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

class PYBIND11_HIDDEN PythonCallableRegistrar {
  DISABLE_COPY_AND_ASSIGN(PythonCallableRegistrar);
  PythonCallableRegistrar() = default;

 public:
  static PythonCallableRegistrar& GetInstance();

  void Register(uint64_t unique_id, const py::object& callable);

  py::object* Get(uint64_t unique_id);

 private:
  std::unordered_map<uint64_t, py::object> python_callable_registry_;
};

void PirCallPythonFunc(py::object* callable,
                       const std::vector<pir::Value>& ins,
                       std::vector<pir::Value>* outs);
}  // namespace pybind
}  // namespace paddle
