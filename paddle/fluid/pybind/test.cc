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

#include "paddle/fluid/pybind/test.h"

#include <Python.h>
#include "paddle/fluid/sub_graph/sub_graph_checker.h"

namespace py = pybind11;

namespace paddle::pybind {
using paddle::test::SubGraphChecker;
using pir::Program;

void BindTest(pybind11::module* module) {
  auto test_module = module->def_submodule("test");
  py::class_<SubGraphChecker, std::shared_ptr<SubGraphChecker>>
      subgraph_checker(
          *test_module,
          "SubGraphChecker",
          R"DOC(A class to check result and speed between PHI and CINN)DOC");
  subgraph_checker
      .def("__init__",
           [](SubGraphChecker& self,
              std::shared_ptr<Program> orig_program,
              std::shared_ptr<Program> prim_program) {
             new (&self) SubGraphChecker(orig_program, prim_program);
           })
      .def("check_result",
           [](std::shared_ptr<SubGraphChecker> self) { self->CheckResult(); })
      .def("check_speed",
           [](std::shared_ptr<SubGraphChecker> self) { self->CheckSpeed(); });
}

}  // namespace paddle::pybind
