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
#include <string>
#include "paddle/fluid/framework/ir/graph.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {
void BindGraph(py::module* m) {
  using paddle::framework::ProgramDesc;
  using paddle::framework::ir::Graph;
  using pybind11::return_value_policy;

  py::class_<Graph>(*m, "Graph", "")
      .def(py::init<const ProgramDesc&>())
      .def("has", &Graph::Has)
      .def("get", &Graph::Get<int>)
      .def("get", &Graph::Get<float>)
      .def("get", &Graph::Get<double>)
      .def("get", &Graph::Get<std::string>)
      .def("set", &Graph::Set<int>)
      .def("set", &Graph::Set<float>)
      .def("set", &Graph::Set<double>)
      .def("set", &Graph::Set<std::string>)
      .def("set_not_owned", &Graph::SetNotOwned<int>)
      .def("set_not_owned", &Graph::SetNotOwned<float>)
      .def("set_not_owned", &Graph::SetNotOwned<double>)
      .def("set_not_owned", &Graph::SetNotOwned<std::string>)
      .def("erase", &Graph::Erase)
      .def("nodes", &Graph::Nodes, return_value_policy::reference)
      .def("create_var_node", &Graph::CreateVarNode,
           return_value_policy::reference)
      .def("create_op_node", &Graph::CreateOpNode,
           return_value_policy::reference)
      .def("create_control_dep_var", &Graph::CreateControlDepVar,
           return_value_policy::reference)
      .def("create_empty_node", &Graph::CreateEmptyNode,
           return_value_policy::reference)
      .def("release_nodes", &Graph::ReleaseNodes)
      .def("remove_node", &Graph::RemoveNode)
      .def("retrieve_node", &Graph::RetrieveNode,
           return_value_policy::reference)
      .def("resolve_hazard", &Graph::ResolveHazard);
}
}  // namespace pybind
}  // namespace paddle
