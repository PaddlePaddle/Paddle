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
#include "paddle/fluid/distributed/fleet_executor/dist_model.h"
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
using paddle::distributed::DistModelConfig;
using paddle::distributed::DistModel;
using paddle::framework::OpDesc;
using paddle::framework::ProgramDesc;

void BindFleetExecutor(py::module* m) {
  py::class_<FleetExecutor>(*m, "FleetExecutor")
      .def(py::init<const std::string&>())
      .def("init", &FleetExecutor::Init)
      .def("run", &FleetExecutor::Run,
           py::call_guard<py::gil_scoped_release>());

  py::class_<TaskNode>(*m, "TaskNode")
      .def(py::init<framework::ProgramDesc*, int64_t, int64_t, int64_t>())
      .def(py::init<int32_t, const std::vector<framework::OpDesc*>&, int64_t,
                    int64_t, int64_t, int64_t>())
      .def("task_id", &TaskNode::task_id)
      .def("add_upstream_task", &TaskNode::AddUpstreamTask)
      .def("add_downstream_task", &TaskNode::AddDownstreamTask)
      .def("set_run_pre_steps", &TaskNode::SetRunPerSteps)
      .def("set_run_at_offset", &TaskNode::SetRunAtOffset)
      .def("set_type", &TaskNode::SetType)
      .def("role", &TaskNode::role)
      .def("init", &TaskNode::Init)
      .def("set_program", &TaskNode::SetProgram);

  py::class_<DistModelConfig>(*m, "DistModelConfig")
      .def(py::init<>())
      .def_readwrite("model_dir", &DistModelConfig::model_dir)
      .def_readwrite("program_desc", &DistModelConfig::program_desc)
      .def_readwrite("scope", &DistModelConfig::scope)
      .def_readwrite("place", &DistModelConfig::place)
      .def_readwrite("device_id", &DistModelConfig::device_id)
      .def_readwrite("trainer_endpoints", &DistModelConfig::trainer_endpoints)
      .def_readwrite("current_endpoint", &DistModelConfig::current_endpoint)
      .def_readwrite("nranks", &DistModelConfig::nranks)
      .def_readwrite("local_rank", &DistModelConfig::local_rank)
      .def_readwrite("mp_degree", &DistModelConfig::mp_degree)
      .def_readwrite("pp_degree", &DistModelConfig::pp_degree)
      .def_readwrite("mp_ring_id", &DistModelConfig::mp_ring_id)
      .def_readwrite("pp_upstream_ring_id",
                     &DistModelConfig::pp_upstream_ring_id)
      .def_readwrite("pp_downstream_ring_id",
                     &DistModelConfig::pp_downstream_ring_id);

  py::class_<DistModel>(*m, "DistModel")
      .def(py::init<const DistModelConfig&>())
      .def("init", &DistModel::Init)
      .def("run", &DistModel::Run, py::call_guard<py::gil_scoped_release>());
}
}  // namespace pybind
}  // namespace paddle
