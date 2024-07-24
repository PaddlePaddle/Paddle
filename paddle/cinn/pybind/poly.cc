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

#include <llvm/Support/FormatVariadic.h>

#include "paddle/cinn/poly/stage.h"
#include "paddle/cinn/pybind/bind.h"
#include "paddle/cinn/pybind/bind_utils.h"

namespace py = pybind11;

namespace cinn::pybind {

using poly::Condition;
using poly::Iterator;
using poly::Stage;
using poly::StageForloopInfo;
using py::arg;

namespace {
void BindMap(py::module *);
void BindStage(py::module *);

void BindMap(py::module *m) {
  py::class_<Iterator> iterator(*m, "Iterator");
  iterator.def_readwrite("id", &Iterator::id)
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def(py::init<const Iterator &>())
      .def("__eq__",
           [](Iterator &self, Iterator &other) { return self == other; })
      .def("__ne__",
           [](Iterator &self, Iterator &other) { return self != other; })
      .def("__str__", [](Iterator &self) { return self.id; })
      .def("__repr__", [](Iterator &self) -> std::string {
        return llvm::formatv("<Iterator {0}>", self.id);
      });

  py::class_<Condition> condition(*m, "Condition");
  condition.def_readwrite("cond", &Condition::cond)
      .def(py::init<std::string>())
      .def("__str__", &Condition::__str__);
}

}  // namespace

void BindPoly(py::module *m) { BindMap(m); }

}  // namespace cinn::pybind
