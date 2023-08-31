// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <string>

#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace py = pybind11;

namespace cinn::pybind {

void BindSchedule(py::module *m) {
  py::class_<ir::IRSchedule> ir_schedule(*m, "IRSchedule");
  ir_schedule
      .def(py::init<const ir::ModuleExpr &,
                    utils::LinearRandomEngine::StateType,
                    bool,
                    utils::ErrorMessageLevel>(),
           py::arg("modexpr"),
           py::arg("rand_seed") = -1,
           py::arg("debug_flag") = false,
           py::arg("err_msg_level") = utils::ErrorMessageLevel::kGeneral)
      .def_static(
          "make",
          [](ir::LoweredFunc &ir_func) {
            ir::ModuleExpr *module_expr = new ir::ModuleExpr({ir_func->body});
            auto scheduler = std::make_unique<ir::IRSchedule>(*module_expr);
            return scheduler;
          })
      .def("fuse",
           py::overload_cast<const std::vector<Expr> &>(&ir::IRSchedule::Fuse))
      .def("get_module",
           py::overload_cast<>(&ir::IRSchedule::GetModule, py::const_))
      .def("get_block",
           py::overload_cast<const std::string &>(&ir::IRSchedule::GetBlock,
                                                  py::const_))
      .def("get_all_blocks",
           py::overload_cast<>(&ir::IRSchedule::GetAllBlocks, py::const_))
      .def("get_loops",
           py::overload_cast<const std::string &>(&ir::IRSchedule::GetLoops,
                                                  py::const_))
      .def("get_name2loops_dict",
           [](const ir::IRSchedule &self, const std::string &block_name) {
             std::vector<ir::Expr> loops = self.GetLoops(block_name);
             std::map<std::string, ir::Expr> name2loops;
             for (const ir::Expr &loop : loops) {
               name2loops[loop.As<ir::For>()->loop_var->name] = loop;
             }
             return name2loops;
           });
}
}  // namespace cinn::pybind
