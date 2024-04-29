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

void BindStageMap(py::module *m) {
  DefineShared<poly::_StageMap_>(m, "StageMap");
  py::class_<poly::StageMap, Shared<poly::_StageMap_>> stage_map(*m,
                                                                 "StageMap");
  stage_map  //
      .def(
          "__getitem__",
          [](poly::StageMap self, ir::Tensor &t) -> Stage & {
            return *self[t];
          },
          py::return_value_policy::reference);

  m->def("create_stages", &poly::CreateStages, py::arg("tensors"));
}

void BindStage(py::module *m) {
  py::class_<Stage> stage(*m, "Stage");
  // enum Stage::ComputeAtKind
  py::enum_<Stage::ComputeAtKind> compute_at_kind(stage, "ComputeAtKind");
  compute_at_kind.value("kComputeAtUnk", Stage::ComputeAtKind::kComputeAtAuto)
      .value("kComputeAtBefore", Stage::ComputeAtKind::kComputeAtBefore)
      .value("kComputeAtAfter", Stage::ComputeAtKind::kComputeAtAfter);

  DefineShared<Stage>(m, "Stage");
  stage.def("id", &Stage::id)
      .def("expr", &Stage::expr)
      .def("axis", py::overload_cast<int>(&Stage::axis, py::const_))
      .def("axis",
           py::overload_cast<const std::string &>(&Stage::axis, py::const_))
      .def("axis_names", &Stage::axis_names)
      .def("bind", &Stage::Bind)
      .def("compute_inline",
           &Stage::ComputeInline,
           "Mark this tensor as inline, and will expand in-place in where it "
           "is used")
      .def(
          "share_buffer_with",
          [](Stage &self, Stage &other) { self.ShareBufferWith(&other); },
          "Share the underlying buffer with another tensor")
      .def("split",
           py::overload_cast<const Iterator &, int>(&Stage::Split),
           arg("level"),
           arg("factor"))
      .def("split",
           py::overload_cast<const std::string &, int>(&Stage::Split),
           arg("level"),
           arg("factor"))
      .def("split",
           py::overload_cast<int, int>(&Stage::Split),
           arg("level"),
           arg("factor"))
      .def("fuse",
           py::overload_cast<int, int>(&Stage::Fuse),
           arg("level0"),
           arg("level1"))
      .def("fuse", py::overload_cast<const std::vector<int> &>(&Stage::Fuse))
      .def("reorder",
           py::overload_cast<const std::vector<Iterator> &>(&Stage::Reorder),
           "Reorder the axis in the computation")
      .def("reorder",
           py::overload_cast<const std::vector<int> &>(&Stage::Reorder),
           "Reorder the axis in the computation")
      .def("tile",
           py::overload_cast<const Iterator &, const Iterator &, int, int>(
               &Stage::Tile))
      .def("tile", py::overload_cast<int, int, int, int>(&Stage::Tile))
      .def("vectorize", py::overload_cast<int, int>(&Stage::Vectorize))
      .def("vectorize",
           py::overload_cast<const std::string &, int>(&Stage::Vectorize))
      .def("vectorize",
           py::overload_cast<const Iterator &, int>(&Stage::Vectorize))
      .def("unroll", py::overload_cast<int>(&Stage::Unroll))
      .def("unroll", py::overload_cast<const std::string &>(&Stage::Unroll))
      .def("unroll", py::overload_cast<const Iterator &>(&Stage::Unroll))
      .def("parallel", py::overload_cast<int>(&Stage::Parallel))
      .def("parallel", py::overload_cast<const std::string &>(&Stage::Parallel))
      .def("parallel", py::overload_cast<const Iterator &>(&Stage::Parallel))
      .def("compute_at",
           &Stage::ComputeAtSchedule,
           arg("other"),
           arg("level"),
           arg("kind") = Stage::kComputeAtAuto)
      .def("skew", &Stage::Skew)
      .def("ctrl_depend", &Stage::CtrlDepend)
      .def("cache_read", &Stage::CacheRead)
      .def("cache_write", &Stage::CacheWrite)
      .def("sync_threads",
           py::overload_cast<poly::StageMap>(&Stage::SyncThreads))
      .def("sync_threads",
           py::overload_cast<int,
                             const std::vector<ir::Tensor> &,
                             poly::StageMap>(&Stage::SyncThreads));
}

}  // namespace

void BindPoly(py::module *m) {
  BindMap(m);
  BindStage(m);
  BindStageMap(m);
}

}  // namespace cinn::pybind
