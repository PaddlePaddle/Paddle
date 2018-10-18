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
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace framework {
namespace ir {

// Node should normally created by Graph::CreateXXXNode().
class Node {
 public:
  enum class Type { kOperation, kVariable };
  static constexpr char kControlDepVarName[] = "__control_var";

  Type NodeType() const { return type_; }

  std::string Name() const { return name_; }

  VarDesc* Var() {
    PADDLE_ENFORCE(IsVar());
    return var_desc_.get();
  }

  OpDesc* Op() const {
    PADDLE_ENFORCE(IsOp());
    return op_desc_.get();
  }

  int id() const { return id_; }

  bool IsOp() const { return type_ == Type::kOperation; }
  bool IsVar() const { return type_ == Type::kVariable; }
  bool IsCtrlVar() const {
    return type_ == Type::kVariable &&
           Name().find(ir::Node::kControlDepVarName) != std::string::npos;
  }

  std::vector<Node*> inputs;
  std::vector<Node*> outputs;

  template <typename AttrType>
  AttrType& Get(const std::string& attr_name) const {
    PADDLE_ENFORCE(Has(attr_name), "%s attr not registered for graph.",
                   attr_name);
    return *boost::any_cast<AttrType*>(attrs_.at(attr_name));
  }

  template <typename AttrType>
  void Set(const std::string& attr_name, AttrType* attr) {
    PADDLE_ENFORCE(attrs_.count(attr_name) == 0, "%s already set in the graph",
                   attr_name);
    attrs_[attr_name] = attr;
    attr_dels_[attr_name] = [attr, attr_name]() {
      VLOG(3) << "deleting " << attr_name;
      delete attr;
    };
  }

  bool Has(const std::string& attr_name) const {
    return attrs_.find(attr_name) != attrs_.end();
  }

 protected:
  const std::string name_;
  std::unique_ptr<VarDesc> var_desc_;
  std::unique_ptr<OpDesc> op_desc_;
  Type type_;
  int id_;

 private:
  // ID can only set by a Graph.
  void SetId(int id) { id_ = id; }

  std::map<std::string, boost::any> attrs_;
  std::map<std::string, std::function<void(void)>> attr_dels_;
  friend class Graph;
  friend std::unique_ptr<Node> CreateNodeForTest(const std::string& name,
                                                 Node::Type type);

  explicit Node(const std::string& name, Type type)
      : name_(name),
        var_desc_(nullptr),
        op_desc_(nullptr),
        type_(type),
        id_(count_++) {}

  explicit Node(VarDesc* var_desc)
      : name_(var_desc->Name()),
        var_desc_(new VarDesc(*var_desc)),
        op_desc_(nullptr),
        type_(Type::kVariable),
        id_(count_++) {}

  explicit Node(OpDesc* op_desc)
      : name_(op_desc->Type()),
        var_desc_(nullptr),
        op_desc_(new OpDesc(*op_desc, op_desc->Block())),
        type_(Type::kOperation),
        id_(count_++) {}

  Node() = delete;

  static int count_;
  static void ResetId() { count_ = 0; }
  DISABLE_COPY_AND_ASSIGN(Node);
};

std::unique_ptr<Node> CreateNodeForTest(const std::string& name,
                                        Node::Type type);

}  // namespace ir
}  // namespace framework
}  // namespace paddle
