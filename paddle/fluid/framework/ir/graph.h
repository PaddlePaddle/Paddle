/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace framework {

namespace details {

// This attr is not recommended, because the graph should not dependence
// the program once it is built.
constexpr char kStaleProgramOpDescs[] = "stale_program_op_descs";
}  //  namespace details

namespace ir {

/*
 * The graph is a Directed Acyclic Single Static Assignment Graph.
 *
 * In more detail, the following properties must hold:
 *
 *   The graph shouldn't contain cycle. Each node is a black-box to the graph
 *   so the node itself could be a loop operator.
 *
 *   Each Variable-type node has only one input (thus single static assignment).
 *
 *   The output/input of operator is variable and the output/input of variable
 *   is operator.
 *
 * The following data harzards in Program are addressed in the Graph:
 *
 *   Write-After-Read
 *     a = op1(x)
 *     x = op2(b)
 *     A control-dependency connection is created bettwen op1 and op2 such that
 *     op1->op2, so as to ensure correct order.
 *
 *   Write-After-Write
 *     x = op1(a)
 *     x = op2(b)
 *     A control-dependency connection is created between op1 and op2 such that
 *     op1->op2, so as to ensure correct order.
 *
 * Other properties currently hold, but is not enforced yet:
 *
 *   Variable-type node (not control dep) with the same variable name share
 *   the same underlying VarDesc.
 */
class Graph {
 public:
  explicit Graph(const ProgramDesc &program);

  virtual ~Graph() {
    for (auto &attr : attrs_) {
      attr_dels_[attr.first]();
    }
    attrs_.clear();
    attr_dels_.clear();
  }

  bool Has(const std::string &attr_name) const {
    return attrs_.count(attr_name) > 0;
  }

  template <typename AttrType>
  AttrType &GetOrInit(const std::string &attr_name) {
    if (!Has(attr_name)) {
      Set(attr_name, new AttrType);
    }
    return Get<AttrType>(attr_name);
  }

  template <typename AttrType>
  AttrType &Get(const std::string &attr_name) const {
    PADDLE_ENFORCE_EQ(Has(attr_name), true, "%s attr not registered for graph.",
                      attr_name);
    try {
      return *boost::any_cast<AttrType *>(attrs_.at(attr_name));
    } catch (boost::bad_any_cast &) {
      PADDLE_THROW(
          "Invalid attribute type of %s error, expected: %s, actual: %s",
          attr_name, typeid(AttrType *).name(),
          attrs_.at(attr_name).type().name());
    }
  }

  template <typename AttrType>
  void Set(const std::string &attr_name, AttrType *attr) {
    PADDLE_ENFORCE_EQ(attrs_.count(attr_name), 0, "%s already set in the graph",
                      attr_name);
    attrs_[attr_name] = attr;
    attr_dels_[attr_name] = [attr, attr_name]() {
      VLOG(3) << "deleting " << attr_name;
      delete attr;
    };
  }

  template <typename AttrType>
  void SetNotOwned(const std::string &attr_name, AttrType *attr) {
    PADDLE_ENFORCE_EQ(attrs_.count(attr_name), 0, "%s already set in the graph",
                      attr_name);
    attrs_[attr_name] = attr;
    attr_dels_[attr_name] = []() {};
  }

  void Erase(const std::string &attr_name) {
    PADDLE_ENFORCE_NE(attrs_.count(attr_name), 0, "%s not set in the graph",
                      attr_name);
    attr_dels_[attr_name]();
    attrs_.erase(attr_name);
    attr_dels_.erase(attr_name);
  }

  const std::unordered_set<ir::Node *> &Nodes() const { return node_set_; }

  // Create a normal variable with non-null VarDesc.
  ir::Node *CreateVarNode(VarDesc *var_desc) {
    PADDLE_ENFORCE_NOT_NULL(var_desc);
    auto *x = AddNode(new ir::Node(var_desc));
    x->SetId(num_node_created_++);
    return x;
  }

  // Create a normal runnable operator with OpDesc.
  ir::Node *CreateOpNode(OpDesc *op_desc) {
    PADDLE_ENFORCE_NOT_NULL(op_desc);
    auto *x = AddNode(new ir::Node(op_desc));
    x->SetId(num_node_created_++);
    return x;
  }

  // Create a control dependency var that connects 2 operations. The
  // var doesn't hold any data. Other than that, it's no different from
  // other var, considering dependency analysis.
  ir::Node *CreateControlDepVar() {
    // TODO(panyx0718): control var name should be really unique.
    const std::string name = string::Sprintf(
        "%s@%llu", static_cast<const char *>(ir::Node::kControlDepVarName),
        num_node_created_);
    auto *x = AddNode(new ir::Node(name, ir::Node::Type::kVariable));
    x->SetId(num_node_created_++);
    return x;
  }

  // A more free style way of creating a graph node. Mostly use for test
  // or "copy" from another node. Avoid using it if possible.
  ir::Node *CreateEmptyNode(const std::string &name, ir::Node::Type type) {
    auto *x = AddNode(new ir::Node(name, type));
    x->SetId(num_node_created_++);
    return x;
  }

  // Clear all node information of the graph and return the ownership of the
  // nodes.
  std::vector<std::unique_ptr<ir::Node>> ReleaseNodes() {
    std::vector<std::unique_ptr<ir::Node>> ret;
    for (auto &n : nodes_) {
      ret.emplace_back(n.second.release());
    }
    nodes_.clear();
    node_set_.clear();
    return ret;
  }

  std::unique_ptr<ir::Node> RemoveNode(ir::Node *node) {
    PADDLE_ENFORCE_EQ(node_set_.find(node) != node_set_.end(), true);
    std::unique_ptr<ir::Node> ret;
    ret.reset(nodes_.at(node).release());
    nodes_.erase(node);
    node_set_.erase(node);
    return ret;
  }

  // NOTE low performance, but simple and secure.
  Node *RetrieveNode(int id) {
    for (auto &node : nodes_) {
      if (node.second->id() == id) {
        return node.second.get();
      }
    }
    return nullptr;
  }

  // Returns reference to the original program.
  // WARN: After a series of passes, the current graph can be quite
  // different from OriginProgram. Caller shouldn't assume much from
  // the returned OriginProgram.
  const ProgramDesc &OriginProgram() const { return program_; }

  // This method takes ownership of `node`.
  ir::Node *AddNode(ir::Node *node) {
    PADDLE_ENFORCE_EQ(node_set_.find(node) == node_set_.end(), true);
    nodes_[node].reset(node);
    node_set_.insert(node);
    return node;
  }

  void ResolveHazard(
      const std::map<std::string, std::vector<ir::Node *>> &var_nodes);

  // Create a new and duplicated graph.
  // WARN: The method only clones the graph structure, not its attributes.
  std::shared_ptr<Graph> Clone();

 private:
  std::map<std::string, std::vector<ir::Node *>> InitFromProgram(
      const ProgramDesc &program);

  // NOTE: program_ shouldn't be exposed to user.
  const ProgramDesc program_;
  std::map<std::string, boost::any> attrs_;
  std::map<std::string, std::function<void(void)>> attr_dels_;
  std::map<ir::Node *, std::unique_ptr<ir::Node>> nodes_;
  std::unordered_set<ir::Node *> node_set_;
  size_t num_node_created_{0};  // help to generate a unique node id.
};

bool IsControlDepVar(const ir::Node &var);
}  // namespace ir
}  // namespace framework
}  // namespace paddle
