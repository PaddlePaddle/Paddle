// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/bind_fleet_executor.h"
#include <pybind11/stl.h>
#include "paddle/fluid/distributed/fleet_executor/fleet_executor.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/place.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

using paddle::distributed::FleetExecutor;
using paddle::distributed::TaskNode;

void BindFleetExecutor(py::module* m) {
  py::class_<FleetExecutor>(*m, "FleetExecutor")
      .def(py::init<const std::string&>())
      .def("init", &FleetExecutor::Init)
      .def("run", &FleetExecutor::Run,
           py::call_guard<py::gil_scoped_release>());

  py::class_<TaskNode>(*m, "TaskNode")
      .def(py::init<const framework::ProgramDesc&, int64_t, int64_t, int64_t>())
      .def(py::init<int32_t, const std::vector<OperatorBase*>&, int64_t,
                    int64_t, int64_t, int64_t>())
      .def("task_id", &TaskNode::task_id)
      .def("add_upstream_task", &TaskNode::AddUpstreamTask)
      .def("add_downstream_task", &TaskNode::AddDownstreamTask)
      .def("set_run_pre_steps", &TaskNode::SetRunPerSteps)
      .def("set_run_at_offset", &TaskNode::SetRunAtOffset)
      .def("set_type", &TaskNode::SetType);
}
}  // namespace pybind
}  // namespace paddle
