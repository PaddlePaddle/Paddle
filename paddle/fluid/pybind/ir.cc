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
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

void BindGraph(py::module* m) {
  using paddle::framework::ProgramDesc;
  using paddle::framework::ir::Graph;
  using paddle::framework::ir::Node;
  using paddle::framework::OpDesc;
  using paddle::framework::VarDesc;
  using pybind11::return_value_policy;

  py::class_<Graph, std::shared_ptr<Graph>>(*m, "Graph", "")
      .def(py::init<const ProgramDesc&>())
      .def("has", &Graph::Has)
      .def("get_int", &Graph::Get<int>)
      .def("get_float", &Graph::Get<float>)
      .def("get_double", &Graph::Get<double>)
      .def("get_string", &Graph::Get<std::string>)
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
      .def("create_var_node",
           [](Graph& self, VarDesc& var_desc) -> std::shared_ptr<Node> {
             return std::shared_ptr<Node>(self.CreateVarNode(&var_desc));
           })
      .def("create_op_node",
           [](Graph& self, OpDesc& op_desc) -> std::shared_ptr<Node> {
             return std::shared_ptr<Node>(self.CreateOpNode(&op_desc));
           })
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

void BindNode(py::module* m) {
  using paddle::framework::ir::Node;
  using paddle::framework::ir::CreateNodeForTest;
  using pybind11::return_value_policy;

  py::class_<Node>(*m, "Node")
      .def(py::init(&CreateNodeForTest), return_value_policy::reference)
      .def("name", &Node::Name)
      .def("node_type", &Node::NodeType)
      .def("var", &Node::Var)
      .def("op", &Node::Op)
      .def("id", &Node::id)
      .def("is_op", &Node::IsOp)
      .def("is_var", &Node::IsVar)
      .def("is_ctrl_var", &Node::IsCtrlVar)
      .def_readwrite("inputs", &Node::inputs)
      .def_readwrite("outputs", &Node::outputs);

  py::enum_<Node::Type>(*m, "NodeType")
      .value("Operation", Node::Type::kOperation)
      .value("Variable", Node::Type::kVariable);
}
}  // namespace pybind
}  // namespace paddle
