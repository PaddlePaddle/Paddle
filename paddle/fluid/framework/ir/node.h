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

class Node {
 public:
  enum class Type { kOperation, kVariable };
  explicit Node(const std::string& name, Type type)
      : name_(name), var_desc_(nullptr), op_desc_(nullptr), type_(type) {}

  explicit Node(VarDesc* var_desc)
      : name_(var_desc->Name()),
        var_desc_(var_desc),
        op_desc_(nullptr),
        type_(Type::kVariable) {}

  explicit Node(OpDesc* op_desc)
      : name_(op_desc->Type()),
        var_desc_(nullptr),
        op_desc_(op_desc),
        type_(Type::kOperation) {}

  Type NodeType() const { return type_; }

  std::string Name() const { return name_; }

  VarDesc* Var() {
    PADDLE_ENFORCE(type_ == Type::kVariable);
    return var_desc_;
  }
  OpDesc* Op() {
    PADDLE_ENFORCE(type_ == Type::kOperation);
    return op_desc_;
  }

  std::vector<Node*> inputs;
  std::vector<Node*> outputs;

 protected:
  const std::string name_;
  VarDesc* var_desc_;
  OpDesc* op_desc_;
  Type type_;

 private:
  DISABLE_COPY_AND_ASSIGN(Node);
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
