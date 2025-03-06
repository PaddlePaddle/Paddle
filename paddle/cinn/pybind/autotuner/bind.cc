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

#include "paddle/cinn/pybind/autotuner/bind.h"

#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/runtime/use_extern_funcs.h"

namespace py = pybind11;

namespace cinn::pybind {

void BindAutotuner(py::module *m) {
  py::module autotuner =
      m->def_submodule("autotuner", "Compiler Infrastructure for Neural Networks");
 
  py::module search = autotuner.def_submodule(
      "search", "namespace cinn::ir::search, Autotuner Search");
  py::module tuner_config = autotuner.def_submodule(
      "tuner_config", "namespace cinn::ir, Autotuner Configuration");


  BindSearch(&search);
  BindTunerConfig(&tuner_config);
}

}  // namespace cinn::pybind
