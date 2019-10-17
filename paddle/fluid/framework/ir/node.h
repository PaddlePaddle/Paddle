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

#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {
namespace ir {

// Node should only created by Graph::CreateXXXNode().
// 1. Every Node should be part of a graph. No dangling Node exists.
// 2. Node only contains members necessary for building graph structure.
//    It doesn't contain other unrelated members, such as device, etc.
//
// Sometimes, for specific usages, Node needs to have additional members,
// such as device_placement, version in order to be executed. It is suggested
// to use composition pattern.
//
// class RunnableOp {
//    RunnableOp(ir::Node* n) : n_(n) { n_.WrappedBy(this); }
//
//    int any_thing_;
// }
//
// RunnableOp is owned by the ir::Node that composes it. In other words.
// ir::Node will be responsible for deleting RunnableOp, say, when ir::Node
// is deleted from the graph.
class Node {
 public:
  virtual ~Node() {
    if (!wrapper_.empty()) {
      VLOG(10) << "ir::Node deleting a wrapper node " << Name();
      wrapper_deleter_();
    }
  }

  enum class Type { kOperation, kVariable };
#if !defined(_WIN32)  // msvc not support constexpr correctly.
  static constexpr char kControlDepVarName[] = "__control_var";
#else
  static const char kControlDepVarName[];
#endif

  Type NodeType() const { return type_; }

  std::string Name() const { return name_; }

  VarDesc* Var() const {
    PADDLE_ENFORCE_EQ(IsVar(), true);
    return var_desc_.get();
  }

  OpDesc* Op() const {
    PADDLE_ENFORCE_EQ(IsOp(), true);
    return op_desc_.get();
  }

  // Set the `wrapper` that wraps the Node. `wrapper` is owned by Node.
  template <typename T>
  void WrappedBy(T* wrapper) {
    if (!wrapper_.empty()) {
      wrapper_deleter_();
    }
    wrapper_ = wrapper;
    wrapper_deleter_ = [wrapper]() { delete wrapper; };
    wrapper_type_ = std::type_index(typeid(T));
  }

  // Return a reference to the `wrapper`.
  template <typename T>
  T& Wrapper() {
    try {
      return *boost::any_cast<T*>(wrapper_);
    } catch (boost::bad_any_cast&) {
      PADDLE_THROW("Invalid wrapper type error, expected %s, actual %s",
                   typeid(T).name(), wrapper_type_.name());
    }
  }

  // Test if the Node is wrapped by type T.
  template <typename T>
  bool IsWrappedBy() const {
    return std::type_index(typeid(T)) == wrapper_type_;
  }

  // Please don't use this API!
  int id() const { return id_; }

  bool IsOp() const { return type_ == Type::kOperation; }
  bool IsVar() const { return type_ == Type::kVariable; }
  bool IsCtrlVar() const {
    return type_ == Type::kVariable &&
           Name().find(ir::Node::kControlDepVarName) != std::string::npos;
  }

  void RenameVar(const std::string& new_name) {
    PADDLE_ENFORCE(type_ == Type::kVariable && var_desc_,
                   "Must be type of variable");
    name_ = new_name;
    var_desc_->SetName(new_name);
  }

  std::vector<Node*> inputs;
  std::vector<Node*> outputs;

 protected:
  std::string name_;
  std::unique_ptr<VarDesc> var_desc_;
  std::unique_ptr<OpDesc> op_desc_;
  Type type_;
  int id_;

 private:
  // ID can only set by a Graph.
  void SetId(int id) { id_ = id; }

  friend class Graph;
  friend std::unique_ptr<Node> CreateNodeForTest(const std::string& name,
                                                 Node::Type type);
  friend std::unique_ptr<Node> CreateNodeForTest(VarDesc* var_desc);
  friend std::unique_ptr<Node> CreateNodeForTest(OpDesc* op_desc);

  explicit Node(const std::string& name, Type type)
      : name_(name), var_desc_(nullptr), op_desc_(nullptr), type_(type) {}

  explicit Node(VarDesc* var_desc)
      : name_(var_desc->Name()),
        var_desc_(new VarDesc(*var_desc)),
        op_desc_(nullptr),
        type_(Type::kVariable) {}

  explicit Node(OpDesc* op_desc)
      : name_(op_desc->Type()),
        var_desc_(nullptr),
        op_desc_(new OpDesc(*op_desc, op_desc->Block())),
        type_(Type::kOperation) {}

  Node() = delete;

  boost::any wrapper_;
  std::function<void(void)> wrapper_deleter_;
  std::type_index wrapper_type_ = std::type_index(typeid(void));

  DISABLE_COPY_AND_ASSIGN(Node);
};

std::unique_ptr<Node> CreateNodeForTest(const std::string& name,
                                        Node::Type type);
std::unique_ptr<Node> CreateNodeForTest(VarDesc* var_desc);

std::unique_ptr<Node> CreateNodeForTest(OpDesc* op_desc);
}  // namespace ir
}  // namespace framework
}  // namespace paddle
