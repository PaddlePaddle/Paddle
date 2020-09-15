// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/compatible.h"

#include <memory>
#include <string>

#include "paddle/fluid/framework/op_version_registry.h"

namespace py = pybind11;

using paddle::framework::compatible::PassVersionCheckerRegistrar;

namespace paddle {
namespace pybind {

void BindCompatible(py::module* m) {
  py::class_<PassVersionCheckerRegistrar>(*m, "PassVersionChecker")
      .def_static("IsCompatible", [](const std::string& name) -> bool {
        auto instance = PassVersionCheckerRegistrar::GetInstance();
        return instance.IsPassCompatible(name);
      });
}

}  // namespace pybind
}  // namespace paddle
