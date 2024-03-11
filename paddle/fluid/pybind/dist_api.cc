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

#include <Python.h>
#include "pybind11/stl.h"

#include "paddle/fluid/pybind/dist_api.h"
#include "paddle/fluid/pybind/dist_static_op_function.h"
#include "paddle/phi/core/enforce.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

void BindDistOpsAPI(pybind11::module *module) {
  {
    if (PyModule_AddFunctions(module->ptr(), DistOpsAPI) < 0) {
      {
        PADDLE_THROW(
            phi::errors::Fatal("Add C++ DistOpsAPI to core.ops failed!"));
      }
    }
  }
}

void BindDistApi(pybind11::module *module) {
  auto ir_module = module->def_submodule("pir");
  auto ops_modules = ir_module.def_submodule("ops");
  BindDistOpsAPI(&ops_modules);
}

}  // namespace pybind
}  // namespace paddle
