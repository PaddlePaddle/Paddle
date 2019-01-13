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
#include <unordered_map>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using paddle::framework::ir::Graph;
using paddle::framework::ir::Node;
using paddle::framework::OpDesc;
using paddle::framework::ProgramDesc;
using paddle::framework::VarDesc;
using pybind11::return_value_policy;

namespace paddle {
namespace pybind {
void BindGraph(py::module *m) {
  py::class_<Graph, std::shared_ptr<Graph>>(
      *m, "Graph",
      "The graph is a Directed Acyclic Single Static Assignment Graph, see "
      "`paddle::ir::Graph` for details.")
      .def(py::init<const ProgramDesc &>())
      .def("has", &Graph::Has)
      .def("get_int", &Graph::Get<int>)
      .def("get_float", &Graph::Get<float>)
      .def("get_double", &Graph::Get<double>)
      .def("get_string", &Graph::Get<std::string>)
      .def("set", [](Graph &self, const std::string &attr_name,
                     int attr) { return self.Set(attr_name, new int(attr)); })
      .def("set",
           [](Graph &self, const std::string &attr_name,
              const std::string &attr) {
             return self.Set(attr_name, new std::string(attr));
           })
      .def("set",
           [](Graph &self, const std::string &attr_name, float attr) {
             return self.Set(attr_name, new float(attr));
           })
      .def("set",
           [](Graph &self, const std::string &attr_name, double attr) {
             return self.Set(attr_name, new double(attr));
           })
      .def("erase", &Graph::Erase)
      .def("nodes", &Graph::Nodes, return_value_policy::reference)
      .def("create_var_node",
           [](Graph &self, VarDesc &var_desc) {
             return self.CreateVarNode(&var_desc);
           },
           return_value_policy::reference)
      .def("create_op_node",
           [](Graph &self, OpDesc &op_desc) {
             return self.CreateOpNode(&op_desc);
           },
           return_value_policy::reference)
      .def("create_control_dep_var", &Graph::CreateControlDepVar,
           return_value_policy::reference)
      .def("create_empty_node", &Graph::CreateEmptyNode,
           return_value_policy::reference)
      .def("release_nodes", &Graph::ReleaseNodes)
      .def("remove_node",
           [](Graph &self, Node &node) { return self.RemoveNode(&node); })
      .def("retrieve_node", &Graph::RetrieveNode,
           return_value_policy::reference)
      .def("resolve_hazard", &Graph::ResolveHazard);
}

void BindNode(py::module *m) {
  py::class_<Node> node(*m, "Node");
  node.def("name", &Node::Name)
      .def("node_type", &Node::NodeType)
      .def("var", &Node::Var)
      .def("op", &Node::Op)
      .def("id", &Node::id)
      .def("is_op", &Node::IsOp)
      .def("is_var", &Node::IsVar)
      .def("is_ctrl_var", &Node::IsCtrlVar)
      .def_readwrite("inputs", &Node::inputs)
      .def_readwrite("outputs", &Node::outputs);

  py::enum_<Node::Type>(node, "Type")
      .value("Operation", Node::Type::kOperation)
      .value("Variable", Node::Type::kVariable)
      .export_values();
}
}  // namespace pybind
}  // namespace paddle
