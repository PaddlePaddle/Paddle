// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <pybind11/detail/common.h>
#include "mlir/IR/Operation.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "Pass/Pass.h"
#include "Pass/PassManager.h"

namespace py = pybind11;

namespace infra {
namespace {

class PassTrampoline : public Pass {
 public:
  using Pass::Pass;

  inline bool CanScheduleOn(mlir::Operation* op) const override {
    PYBIND11_OVERRIDE(bool, Pass, CanScheduleOn, op);
  }
  void Run(mlir::Operation* op) override {
    PYBIND11_OVERRIDE_PURE(void, Pass, Run, op);
  }
};

class PassPublicist : public Pass {
 public:
  using Pass::CanScheduleOn;
  using Pass::Pass;
  using Pass::Run;
};
}  // namespace
PYBIND11_MODULE(pass_python, m) {
  py::class_<PassInfo>(m, "PassInfo")
      .def(py::init<const std::string&, int, const std::vector<std::string>&>())
      .def_readwrite("name", &PassInfo::name, "...")
      .def_readwrite("opt_level", &PassInfo::opt_level, "...")
      .def_readwrite("dependents", &PassInfo::dependents, "...");

  py::class_<Pass, PassTrampoline>(m, "Pass")
      .def(py::init<const std::string&, int, const std::vector<std::string>&>())
      .def("can_schedule_on", &PassPublicist::CanScheduleOn, "...")
      .def("run", &PassPublicist::Run, "...")
      .def("get_pass_info", &Pass::GetPassInfo, "...");

  py::class_<PassManager>(m, "PassManager")
      .def(py::init<mlir::MLIRContext*, int>())
      .def("run", &PassManager::Run)

      // pybind not support `RetType func(unique_ptr)` signature.
      // https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html#std-unique-ptr
      .def("add_pass", [](PassManager& pm, Pass* pass) {
        auto cp = pass->Clone();
        pm.addPass(std::move(cp));
      });
}
}  // namespace infra
