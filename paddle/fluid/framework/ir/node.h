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

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/platform/macros.h"

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

  // Please don't use this API!
  int id() const { return id_; }

  bool IsOp() const { return type_ == Type::kOperation; }
  bool IsVar() const { return type_ == Type::kVariable; }
  bool IsCtrlVar() const {
    return type_ == Type::kVariable &&
           Name().find(ir::Node::kControlDepVarName) != std::string::npos;
  }

  std::vector<Node*> inputs;
  std::vector<Node*> outputs;

 protected:
  const std::string name_;
  std::unique_ptr<VarDesc> var_desc_;
  std::unique_ptr<OpDesc> op_desc_;
  Type type_;
  int id_;

 private:
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
  // Please don't use this API or make this public.
  static void ResetId() { count_ = 0; }
  DISABLE_COPY_AND_ASSIGN(Node);
};

std::unique_ptr<Node> CreateNodeForTest(const std::string& name,
                                        Node::Type type);

}  // namespace ir
}  // namespace framework
}  // namespace paddle
