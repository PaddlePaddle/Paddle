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

#include "paddle/fluid/pybind/ir.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "paddle/ir/core/block.h"
#include "paddle/ir/core/program.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using ir::Block;
using ir::Operation;
using ir::Program;
using pybind11::return_value_policy;

namespace paddle {
namespace pybind {

void BindProgram(py::module *m) {
  py::class_<Program> program(*m, "Program");
  program.def("parameters_num", &Program::parameters_num)
      .def("block",
           py::overload_cast<>(&Program::block),
           return_value_policy::reference)
      .def("block",
           py::overload_cast<>(&Program::block, py::const_),
           return_value_policy::reference)
      .def("print", [](Program &self) {
        std::ostringstream print_stream;
        self.Print(print_stream);
        LOG(INFO) << print_stream.str();
      });
}

void BindBlock(py::module *m) {
  py::class_<Block> block(*m, "Block");
  block.def("front", &Block::front, return_value_policy::reference)
      .def("get_op_list", [](Block &self) -> py::list {
        py::list op_list;
        for (auto iter = self.begin(); iter != self.end(); iter++) {
          op_list.append(*iter);
        }
        return op_list;
      });
}

void BindOperation(py::module *m) {
  py::class_<Operation> op(*m, "Operation");
  op.def("name", &Operation::name);
}

void BindNewIR(pybind11::module *m) {
  BindProgram(m);
  BindBlock(m);
  BindOperation(m);
}

}  // namespace pybind
}  // namespace paddle
