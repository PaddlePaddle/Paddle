// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <pybind11/functional.h>

#include <functional>

#include "paddle/cinn/backends/compiler.h"
#include "paddle/cinn/backends/llvm/execution_engine.h"
#include "paddle/cinn/pybind/bind.h"

namespace py = pybind11;

struct cinn_pod_value_t;

namespace cinn::pybind {

using backends::Compiler;
using backends::ExecutionEngine;
using backends::ExecutionOptions;

namespace {

void BindExecutionEngine(py::module *);

void BindExecutionEngine(py::module *m) {
  py::class_<ExecutionOptions> options(*m, "ExecutionOptions");
  options.def(py::init<>())
      .def_readwrite("opt_level", &ExecutionOptions::opt_level)
      .def_readwrite("enable_debug_info", &ExecutionOptions::enable_debug_info);

  auto lookup = [](ExecutionEngine &self, absl::string_view name) {
    auto *function_ptr =
        reinterpret_cast<void (*)(void **, int32_t)>(self.Lookup(name));
    auto function_wrapper =
        [function_ptr](std::vector<cinn_pod_value_t> &args) {
          function_ptr(reinterpret_cast<void **>(args.data()), args.size());
        };
    return std::function<void(std::vector<cinn_pod_value_t> &)>(
        function_wrapper);
  };

  py::class_<ExecutionEngine> engine(*m, "ExecutionEngine");
  engine
      .def_static(
          "create",
          py::overload_cast<const ExecutionOptions &>(&ExecutionEngine::Create),
          py::arg("options") = ExecutionOptions())
      .def(py::init(py::overload_cast<const ExecutionOptions &>(
               &ExecutionEngine::Create)),
           py::arg("options") = ExecutionOptions())
      .def("lookup", lookup)
      .def("link", &ExecutionEngine::Link, py::arg("module"));

  {
    auto lookup = [](Compiler &self, absl::string_view name) {
      auto *function_ptr =
          reinterpret_cast<void (*)(void **, int32_t)>(self.Lookup(name));
      auto function_wrapper =
          [function_ptr](std::vector<cinn_pod_value_t> &args) {
            function_ptr(reinterpret_cast<void **>(args.data()), args.size());
          };
      return std::function<void(std::vector<cinn_pod_value_t> &)>(
          function_wrapper);
    };

    py::class_<Compiler> compiler(*m, "Compiler");
    compiler
        .def_static("create", &Compiler::Create)  //
        .def("build", &Compiler::BuildDefault)    //
        .def("lookup", lookup);
  }
}

}  // namespace

void BindBackends(py::module *m) { BindExecutionEngine(m); }
}  // namespace cinn::pybind
