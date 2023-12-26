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
                    utils::ErrorMessageLevel,
                    bool>(),
           py::arg("modexpr"),
           py::arg("rand_seed") = -1,
           py::arg("debug_flag") = false,
           py::arg("err_msg_level") = utils::ErrorMessageLevel::kGeneral,
           py::arg("is_dynamic_shape") = false)
      .def_static(
          "make",
          [](ir::LoweredFunc &ir_func) {
            ir::ModuleExpr *module_expr = new ir::ModuleExpr({ir_func->body});
            auto scheduler = std::make_unique<ir::IRSchedule>(
                *module_expr,
                /* rand_seed = */ -1,
                /* debug_flag = */ false,
                /* err_msg_level = */ utils::ErrorMessageLevel::kGeneral,
                /* is_dynamic_shape = */ true);
            return scheduler;
          })
      .def("fuse",
           py::overload_cast<const std::vector<Expr> &>(&ir::IRSchedule::Fuse))
      .def("split",
           py::overload_cast<const Expr &, const std::vector<int> &>(
               &ir::IRSchedule::Split),
           py::arg("loop"),
           py::arg("factors"))
      .def("compute_at",
           py::overload_cast<const Expr &, const Expr &, bool>(
               &ir::IRSchedule::ComputeAt),
           py::arg("block"),
           py::arg("loop"),
           py::arg("keep_unit_loops") = false)
      .def("simple_compute_at",
           py::overload_cast<const Expr &, const Expr &>(
               &ir::IRSchedule::SimpleComputeAt),
           py::arg("block"),
           py::arg("loop"))
      .def("reverse_compute_at",
           py::overload_cast<const Expr &, const Expr &, bool>(
               &ir::IRSchedule::ReverseComputeAt),
           py::arg("block"),
           py::arg("loop"),
           py::arg("keep_unit_loops") = false)
      .def("cache_read",
           py::overload_cast<const Expr &, int, const std::string &>(
               &ir::IRSchedule::CacheRead))
      .def("cache_write",
           py::overload_cast<const Expr &, int, const std::string &>(
               &ir::IRSchedule::CacheWrite))
      .def("sync_threads",
           py::overload_cast<const Expr &, bool>(&ir::IRSchedule::SyncThreads),
           py::arg("ir_node"),
           py::arg("after_node") = true)
      .def("set_buffer",
           py::overload_cast<Expr &, const std::string &, bool>(
               &ir::IRSchedule::SetBuffer),
           py::arg("block"),
           py::arg("memory_type"),
           py::arg("fixed") = false)
      .def("reorder",
           py::overload_cast<const std::vector<Expr> &>(
               &ir::IRSchedule::Reorder))
      .def("parallel",
           py::overload_cast<const Expr &>(&ir::IRSchedule::Parallel))
      .def("vectorize",
           py::overload_cast<const Expr &, int>(&ir::IRSchedule::Vectorize))
      .def("unroll", py::overload_cast<const Expr &>(&ir::IRSchedule::Unroll))

      .def("compute_inline",
           py::overload_cast<const Expr &>(&ir::IRSchedule::ComputeInline))
      .def("reverse_compute_inline",
           py::overload_cast<const Expr &>(
               &ir::IRSchedule::ReverseComputeInline))
      .def("bind", &ir::IRSchedule::Bind)
      .def("copy_transform_and_loop_info",
           py::overload_cast<const Expr &, const Expr &>(
               &ir::IRSchedule::CopyTransformAndLoopInfo))
      .def("rfactor",
           py::overload_cast<const Expr &, int>(&ir::IRSchedule::Rfactor))
      .def("annotate",
           py::overload_cast<const Expr &,
                             const std::string &,
                             const ir::attr_t &>(&ir::IRSchedule::Annotate))
      .def("unannotate",
           py::overload_cast<Expr &, const std::string &>(
               &ir::IRSchedule::Unannotate))
      .def("flatten_loops",
           py::overload_cast<const std::vector<Expr> &, const bool>(
               &ir::IRSchedule::FlattenLoops),
           py::arg("loops"),
           py::arg("force_flat") = false)
      .def("sample_perfect_tile",
           py::overload_cast<const Expr &, int, int, const std::vector<int> &>(
               &ir::IRSchedule::SamplePerfectTile),
           py::arg("loop"),
           py::arg("n"),
           py::arg("max_innermost_factor"),
           py::arg("decision") = std::vector<int>())
      .def("sample_categorical",
           py::overload_cast<const std::vector<int> &,
                             const std::vector<float> &,
                             const std::vector<int> &>(
               &ir::IRSchedule::SampleCategorical),
           py::arg("candidates"),
           py::arg("probs"),
           py::arg("decision") = std::vector<int>())
      .def("get_module",
           py::overload_cast<>(&ir::IRSchedule::GetModule, py::const_))
      .def("get_root_block", &ir::IRSchedule::GetRootBlock)
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
