/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <Python.h>
#include <paddle/framework/scope.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace pd = paddle::framework;

PYBIND11_PLUGIN(core) {
  py::module m("core", "C++ core of Paddle Paddle");

  py::class_<pd::Variable>(m, "Variable", R"DOC(Variable Class.

All parameter, weight, gradient are variables in Paddle.
)DOC")
      .def("is_int", [](const pd::Variable& var) { return var.IsType<int>(); })
      .def("set_int",
           [](pd::Variable& var, int val) -> void {
             *var.GetMutable<int>() = val;
           })
      .def("get_int",
           [](const pd::Variable& var) -> int { return var.Get<int>(); });

  py::class_<pd::Scope, std::shared_ptr<pd::Scope>>(m, "Scope")
      .def(py::init<const std::shared_ptr<pd::Scope>&>())
      .def("get_var",
           &pd::Scope::GetVariable,
           py::return_value_policy::reference)
      .def("create_var",
           &pd::Scope::CreateVariable,
           py::return_value_policy::reference);

  return m.ptr();
}
