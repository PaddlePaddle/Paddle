// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <list>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_lite.h"

namespace paddle {
namespace lite {
namespace mir {

// Node in a MIR graph.
class Node {
 public:
  std::list<Node*> inlinks;
  std::list<Node*> outlinks;

  Node() = default;

  enum class Role {
    kArg = 0,
    kStmt,
    kNumRoles, /*should be last*/
    kUnk,
  };

  struct Stmt {
    std::string op_type;
    // The kernel instances this Statement contains.
    std::vector<std::unique_ptr<KernelBase>> valid_kernels;
    // TODO(Superjomn) make this a shared_ptr for resource safety.
    std::shared_ptr<OpLite> op;  // we hold op to run InferShape

    const OpInfo* op_info() {
      CHECK(op);
      return op->op_info();
    }

    Place place() const {
      CHECK(!valid_kernels.empty());
      return valid_kernels.front()->place();
    }

    KernelBase& picked_kernel() {
      CHECK(!valid_kernels.empty()) << "no kernel for " << op_type;
      return *valid_kernels.front();
    }

    friend std::ostream& operator<<(std::ostream& os, const Stmt& other) {
      os << "Statement " << other.op_type << " " << other.place();
      return os;
    }
  };

  struct Arg {
    std::string name;
    const Type* type{};
    // Weight is a special kind of argument, it is marked as weight explicitly
    // so that some weight related optimization can take place.
    bool is_weight{false};
  };

  Arg& AsArg(const std::string& name) {
    auto& x = AsArg();
    x.name = name;
    return x;
  }

  Stmt& AsStmt(const std::string& op_type,
               std::vector<std::unique_ptr<KernelBase>>&& kernels,
               const std::shared_ptr<OpLite>& op) {
    auto& x = AsStmt();
    x.op_type = op_type;
    x.op = op;
    x.valid_kernels = std::move(kernels);
    return x;
  }

  // Set roles.
  Arg& AsArg() {
    if (role_ != Role::kUnk) {
      CHECK(role_ == Role::kArg);
      return *arg_;
    }
    role_ = Role::kArg;
    arg_.reset(new Arg);
    return *arg_;
  }
  Stmt& AsStmt() {
    if (role_ != Role::kUnk) {
      CHECK(role_ == Role::kStmt);
      return *stmt_;
    }
    role_ = Role::kStmt;
    stmt_.reset(new Stmt);
    return *stmt_;
  }

  friend std::ostream& operator<<(std::ostream& os, Node& other) {
    os << static_cast<int>(other.role_) << " ";
    if (!other.IsRoleSet()) {
      os << "Unk role node";
    }
    if (other.IsArg()) {
      auto& arg = other.AsArg();
      os << "Argument " << arg.name;
    }
    if (other.IsStmt()) {
      auto& arg = other.AsStmt();
      os << "Statement " << arg.op_type;
    }
    return os;
  }

  // Check roles.
  bool IsRoleSet() const { return role_ != Role::kUnk; }
  bool IsStmt() const { return role_ == Role::kStmt; }
  bool IsArg() const { return role_ == Role::kArg; }

 private:
  // Either stmt_ or argument_ is used.
  std::unique_ptr<Stmt> stmt_;
  std::unique_ptr<Arg> arg_;

  Role role_{Role::kUnk};
};
}  // namespace mir
}  // namespace lite
}  // namespace paddle
