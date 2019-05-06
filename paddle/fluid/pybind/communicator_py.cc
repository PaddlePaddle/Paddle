/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pybind/communicator_py.h"

#include <Python.h>

#include "paddle/fluid/framework/program_desc.h"
#include "pybind11/pybind11.h"

#include "paddle/fluid/operators/distributed/communicator.h"

namespace py = pybind11;

using paddle::framework::ProgramDesc;
using paddle::operators::distributed::Communicator;
using paddle::framework::Scope;

namespace paddle {
namespace pybind {

void BindCommunicator(py::module* m) {
  py::class_<Communicator, std::shared_ptr<Communicator>>(*m, "communicator")
      .def(py::init([](const ProgramDesc& program, Scope* param_scope) {
        Communicator::Init(program, param_scope);
        return Communicator::GetInstantcePtr();
      }))
      .def("start", &Communicator::Start);
}

}  // namespace pybind
}  // namespace paddle
