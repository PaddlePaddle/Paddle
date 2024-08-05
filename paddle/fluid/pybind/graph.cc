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

#include "paddle/fluid/pybind/graph.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/python_headers.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_desc.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using paddle::framework::OpDesc;
using paddle::framework::ProgramDesc;
using paddle::framework::Scope;
using paddle::framework::VarDesc;
using paddle::framework::ir::BuildOperationAdjList;
using paddle::framework::ir::Graph;
using paddle::framework::ir::GraphNum;
using paddle::framework::ir::GraphSafeRemoveNodes;
using paddle::framework::ir::HasCircle;
using paddle::framework::ir::Node;
using paddle::framework::ir::NodeComp;
using paddle::framework::ir::TopologySortOperations;
using pybind11::return_value_policy;

namespace paddle::pybind {
void BindGraph(py::module *m) {
  m->def("graph_safe_remove_nodes",
         [](Graph *graph, const std::unordered_set<const Node *> &nodes) {
           return GraphSafeRemoveNodes(graph, nodes);
         });
  m->def("has_circle", HasCircle);
  m->def("graph_num", GraphNum);
  m->def(
      "topology_sort", TopologySortOperations, return_value_policy::reference);
  m->def("build_adjacency_list",
         BuildOperationAdjList<NodeComp>,
         return_value_policy::reference);
  py::class_<Graph, std::shared_ptr<Graph>>(
      *m,
      "Graph",
      "The graph is a Directed Acyclic Single Static Assignment Graph, see "
      "`paddle::ir::Graph` for details.")
      .def(py::init<const ProgramDesc &>())
      .def(py::init<const ProgramDesc &, int64_t, int64_t>())
      .def("clone", &Graph::Clone)
      .def("has", &Graph::Has)
      .def("get_bool", &Graph::Get<bool>)
      .def("get_int", &Graph::Get<int>)
      .def("get_float", &Graph::Get<float>)
      .def("get_double", &Graph::Get<double>)
      .def("get_string", &Graph::Get<std::string>)
      .def("get_marked_nodes",
           &Graph::Get<std::unordered_set<const Node *>>,
           return_value_policy::reference)
      .def("set",
           [](Graph &self, const std::string &attr_name, bool attr) {
             return self.Set(attr_name, new bool(attr));
           })
      .def("set",
           [](Graph &self, const std::string &attr_name, int attr) {
             return self.Set(attr_name, new int(attr));
           })
      .def("set",
           [](Graph &self,
              const std::string &attr_name,
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
      .def("set",
           [](Graph &self,
              const std::string &attr_name,
              const std::unordered_set<const Node *> &attr) {
             return self.Set(attr_name,
                             new std::unordered_set<const Node *>(attr));
           })
      .def("set",
           [](Graph &self,
              const std::string &attr_name,
              const std::unordered_set<std::string> &attr) {
             return self.Set(attr_name,
                             new std::unordered_set<std::string>(attr));
           })
      .def("set_not_owned",
           [](Graph &self, const std::string &attr_name, Scope &attr) {
             self.SetNotOwned<Scope>(attr_name, &attr);
           })
      .def("erase", &Graph::Erase)
      .def("nodes", &Graph::Nodes, return_value_policy::reference)
      .def(
          "create_var_node",
          [](Graph &self, VarDesc &var_desc) {
            return self.CreateVarNode(&var_desc);
          },
          return_value_policy::reference)
      .def(
          "create_op_node",
          [](Graph &self, OpDesc &op_desc) {
            return self.CreateOpNode(&op_desc);
          },
          return_value_policy::reference)
      .def("create_control_dep_var",
           &Graph::CreateControlDepVar,
           return_value_policy::reference)
      .def("create_empty_node",
           &Graph::CreateEmptyNode,
           return_value_policy::reference)
      .def("release_nodes", &Graph::ReleaseNodes)
      .def("remove_node",
           [](Graph &self, Node &node) { return self.RemoveNode(&node); })
      .def(
          "retrieve_node", &Graph::RetrieveNode, return_value_policy::reference)
      .def("resolve_hazard", &Graph::ResolveHazard)
      .def("origin_program_desc",
           &Graph::OriginProgram,
           return_value_policy::reference)
      .def("sub_graph_size", &Graph::SubGraphsSize)
      .def("get_sub_graph", [](Graph &self, int i) {
        /* Here we use a lambda function as an empty deleter to avoid the double
        free of smart pointer.
        Otherwise, this shared pointer will be free both in python and
        cpp scope, which will lead a core dumped. */
        return std::shared_ptr<Graph>(self.GetSubGraph(i), [](Graph *) {});
      });
}

void BindNode(py::module *m) {
  py::class_<Node> node(*m, "Node");
  node.def("name", &Node::Name)
      .def("node_type", &Node::NodeType)
      .def("var", &Node::Var, return_value_policy::reference)
      .def("op", &Node::Op, return_value_policy::reference)
      .def("id", &Node::id)
      .def("graph_id", &Node::GraphId)
      .def("original_desc_id", &Node::OriginalDescId)
      .def("is_op", &Node::IsOp)
      .def("is_var", &Node::IsVar)
      .def("is_ctrl_var", &Node::IsCtrlVar)
      .def("clear_inputs", [](Node &self) { self.inputs.clear(); })
      .def("remove_input",
           [](Node &self, int node_id) {
             auto pos = std::find_if(
                 self.inputs.begin(),
                 self.inputs.end(),
                 [&node_id](const Node *n) { return n->id() == node_id; });
             if (pos != self.inputs.end()) {
               self.inputs.erase(pos);
             }
           })
      .def("remove_input",
           [](Node &self, Node &node) {
             auto pos =
                 std::find(self.inputs.begin(), self.inputs.end(), &node);
             if (pos != self.inputs.end()) {
               self.inputs.erase(pos);
             }
           })
      .def("append_input",
           [](Node &self, Node &node) { self.inputs.push_back(&node); })
      .def("clear_outputs", [](Node &self) { self.outputs.clear(); })
      .def("remove_output",
           [](Node &self, int node_id) {
             auto pos = std::find_if(
                 self.outputs.begin(),
                 self.outputs.end(),
                 [&node_id](const Node *n) { return n->id() == node_id; });
             if (pos != self.outputs.end()) {
               self.outputs.erase(pos);
             }
           })
      .def("remove_output",
           [](Node &self, Node &node) {
             auto pos =
                 std::find(self.outputs.begin(), self.outputs.end(), &node);
             if (pos != self.outputs.end()) {
               self.outputs.erase(pos);
             }
           })
      .def("append_output",
           [](Node &self, Node &node) { self.outputs.push_back(&node); })
      .def_readwrite("inputs", &Node::inputs)
      .def_readwrite("outputs", &Node::outputs);

  py::enum_<Node::Type>(node, "Type")
      .value("Operation", Node::Type::kOperation)
      .value("Variable", Node::Type::kVariable)
      .export_values();

  py::enum_<Node::Dep>(node, "Dep")
      .value("Same", Node::Dep::kSame)
      .value("Before", Node::Dep::kBefore)
      .value("After", Node::Dep::kAfter)
      .value("NoDep", Node::Dep::kNoDep)
      .export_values();
}

class PYBIND11_HIDDEN PassAttrGetterSetterRegistry {
 private:
  PassAttrGetterSetterRegistry() : getter_setter_map_() {}
  DISABLE_COPY_AND_ASSIGN(PassAttrGetterSetterRegistry);

  using Getter = std::function<py::object(const framework::ir::Pass & /*pass*/,
                                          const std::string & /*attr_name*/)>;
  using Setter = std::function<void(const std::string & /*attr_name*/,
                                    const py::object & /*attr_value*/,
                                    framework::ir::Pass * /*pass*/)>;

  struct GetterSetter {
    Getter getter;
    Setter setter;
  };

 public:
  static PassAttrGetterSetterRegistry &Instance() {
    static PassAttrGetterSetterRegistry instance;
    return instance;
  }

  void Register(const std::string &attr_type, Getter getter, Setter setter) {
    PADDLE_ENFORCE_NOT_NULL(
        getter,
        common::errors::InvalidArgument("getter of %s should not be nullptr",
                                        attr_type));
    PADDLE_ENFORCE_NOT_NULL(
        setter,
        common::errors::InvalidArgument("setter of %s should not be nullptr",
                                        attr_type));
    GetterSetter getter_setter;
    getter_setter.getter = std::move(getter);
    getter_setter.setter = std::move(setter);
    PADDLE_ENFORCE_EQ(
        getter_setter_map_.emplace(attr_type, getter_setter).second,
        true,
        common::errors::InvalidArgument(
            "getter and setter of %s have been set before", attr_type));
  }

  py::object Get(const framework::ir::Pass &pass,
                 const std::string &attr_name,
                 const std::string &attr_type) const {
    auto iter = getter_setter_map_.find(attr_type);
    PADDLE_ENFORCE_EQ(
        iter != getter_setter_map_.end(),
        true,
        common::errors::InvalidArgument(
            "unsupported attribute type %s of %s", attr_type, attr_name));
    const auto &getter = iter->second.getter;
    return getter(pass, attr_name);
  }

  void Set(const std::string &attr_name,
           const std::string &attr_type,
           const py::object &attr_value,
           framework::ir::Pass *pass) const {
    auto iter = getter_setter_map_.find(attr_type);
    PADDLE_ENFORCE_EQ(
        iter != getter_setter_map_.end(),
        true,
        common::errors::InvalidArgument(
            "unsupported attribute type %s of %s", attr_type, attr_name));
    const auto &setter = iter->second.setter;
    setter(attr_name, attr_value, pass);
  }

 private:
  std::unordered_map<std::string, GetterSetter> getter_setter_map_;
};

#define REGISTER_PASS_ATTR_GETTER_SETTER(attr_type_name, cpp_type)           \
  do {                                                                       \
    auto getter = [](const framework::ir::Pass &pass,                        \
                     const std::string &attr_name) -> py::object {           \
      auto attr_value = pass.Get<cpp_type>(attr_name);                       \
      return py::cast(attr_value);                                           \
    };                                                                       \
    auto setter = [](const std::string &attr_name,                           \
                     const py::object &attr_value,                           \
                     framework::ir::Pass *pass) {                            \
      PADDLE_ENFORCE_NOT_NULL(                                               \
          pass, common::errors::InvalidArgument("pass should be provided")); \
      try {                                                                  \
        const auto &cpp_attr_value = py::cast<cpp_type>(attr_value);         \
        pass->Set(attr_name, new cpp_type(cpp_attr_value));                  \
      } catch (py::cast_error &) {                                           \
        PADDLE_THROW(common::errors::InvalidArgument(                        \
            "type error of attribute %s, expected to be %s",                 \
            attr_name,                                                       \
            attr_type_name));                                                \
      }                                                                      \
    };                                                                       \
    PassAttrGetterSetterRegistry::Instance().Register(                       \
        attr_type_name, getter, setter);                                     \
  } while (0)

// NOTE: attr_types may be changed
static void SetAttrsToPass(
    const std::unordered_map<std::string, py::object> &attrs,
    std::unordered_map<std::string, std::string> *attr_types,
    framework::ir::Pass *pass) {
  for (const auto &name_and_value : attrs) {
    const auto &attr_name = name_and_value.first;
    const auto &attr_value = name_and_value.second;
    auto &attr_type = (*attr_types)[attr_name];
    if (attr_type.empty()) {
      attr_type = py::cast<std::string>(attr_value.get_type().attr("__name__"));
    }
    PassAttrGetterSetterRegistry::Instance().Set(
        attr_name, attr_type, attr_value, pass);
  }
}

static std::vector<std::string> GetPassNames(const py::object &names) {
  try {
    return {py::cast<std::string>(names)};
  } catch (py::cast_error &) {
    try {
      return py::cast<std::vector<std::string>>(names);
    } catch (py::cast_error &) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Pass names must be either str or list[str]"));
    }
  }
}

void BindPass(py::module *m) {
  // NOTE: pass_attr_types is a dict to indicate the type of each attribute.
  // Python has only one integral type "int", but C++ has many integral types.
  // If pass_attrs = {"nranks": 1} in Python, we cannot know whether the type
  // of "nranks" is size_t or int in C++. Therefore, users can set
  // pass_attr_types to indicate the type of "nranks" explicitly,
  // i.e. pass_attr_types = {"nranks": "size_t"} means that the type of
  // "nranks" is size_t in C++.
  REGISTER_PASS_ATTR_GETTER_SETTER("bool", bool);
  REGISTER_PASS_ATTR_GETTER_SETTER("int", int64_t);
  REGISTER_PASS_ATTR_GETTER_SETTER("long", int64_t);
  REGISTER_PASS_ATTR_GETTER_SETTER("size_t", size_t);
  REGISTER_PASS_ATTR_GETTER_SETTER("float32", float);
  // Python float is C++ double
  REGISTER_PASS_ATTR_GETTER_SETTER("float", double);
  REGISTER_PASS_ATTR_GETTER_SETTER("bytes", std::string);
  REGISTER_PASS_ATTR_GETTER_SETTER("str", std::string);
  REGISTER_PASS_ATTR_GETTER_SETTER("list[str]", std::vector<std::string>);

  m->def("apply_pass",
         [](framework::ProgramDesc *main_program,
            framework::ProgramDesc *startup_program,
            const py::object &py_pass_names,
            const std::unordered_map<std::string, py::object> &pass_attrs,
            std::unordered_map<std::string, std::string> pass_attr_types) {
           auto pass_names = GetPassNames(py_pass_names);
           std::vector<std::unique_ptr<framework::ir::Pass>> passes;
           std::vector<const framework::ir::Pass *> passes_not_owned;
           passes.reserve(pass_names.size());
           passes_not_owned.reserve(pass_names.size());
           for (const auto &name : pass_names) {
             auto pass = framework::ir::PassRegistry::Instance().Get(name);
             SetAttrsToPass(pass_attrs, &pass_attr_types, pass.get());
             passes.push_back(std::move(pass));
             passes_not_owned.push_back(passes.back().get());
           }

           framework::ir::Pass::ApplyPassesToProgram(
               passes_not_owned, main_program, startup_program);
           std::unordered_map<std::string, py::object> result_attrs;
           for (const auto &pass : passes) {
             for (const auto &name_and_value : pass_attrs) {
               const auto &attr_name = name_and_value.first;
               const auto &attr_type = pass_attr_types.at(attr_name);
               result_attrs[attr_name] =
                   PassAttrGetterSetterRegistry::Instance().Get(
                       *pass, attr_name, attr_type);
             }
           }
           return result_attrs;
         });
}

}  // namespace paddle::pybind
