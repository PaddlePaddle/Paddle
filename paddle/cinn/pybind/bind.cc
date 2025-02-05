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

#include "paddle/cinn/pybind/bind.h"

#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/runtime/use_extern_funcs.h"

namespace py = pybind11;

namespace cinn::pybind {

void BindCINN(py::module *m) {
  py::module cinn =
      m->def_submodule("cinn", "Compiler Infrastructure for Neural Networks");
  py::module runtime = cinn.def_submodule("runtime", "bind cinn_runtime");
  py::module common = cinn.def_submodule("common", "namespace cinn::common");
  py::module lang = cinn.def_submodule("lang", "namespace cinn::lang");
  py::module ir = cinn.def_submodule("ir", "namespace cinn::ir");
  py::module poly =
      cinn.def_submodule("poly", "namespace cinn::poly, polyhedral");
  py::module backends = cinn.def_submodule(
      "backends", "namespace cinn::backends, execution backends");
  py::module optim = cinn.def_submodule(
      "optim", "namespace cinn::optim, CINN IR optimization");
  py::module pe = cinn.def_submodule(
      "pe", "namespace cinn::hlir::pe, CINN Primitive Emitters");
  py::module frontend =
      cinn.def_submodule("frontend", "namespace cinn::frontend, CINN frontend");
  py::module framework = cinn.def_submodule(
      "framework", "namespace cinn::hlir::framework, CINN framework");
  py::module utils =
      cinn.def_submodule("utils", "namespace cinn::utils, CINN framework");
  py::module schedule = cinn.def_submodule(
      "schedule", "namespace cinn::ir::schedule, CINN Schedule");

  BindRuntime(&runtime);
  BindCommon(&common);
  BindIr(&ir);
  BindLang(&lang);
  BindPoly(&poly);
  BindBackends(&backends);
  BindOptim(&optim);
  BindPE(&pe);
  BindFramework(&framework);
  BindUtils(&utils);
  BindSchedule(&schedule);
}

}  // namespace cinn::pybind
