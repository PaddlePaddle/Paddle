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
    kArgument = 0,
    kInstruct,
    kNumRoles, /*should be last*/
    kUnk,
  };

  struct Instruct {
    std::string op_type;
    // The kernel instances this Instruct contains.
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
      CHECK(!valid_kernels.empty());
      return *valid_kernels.front();
    }

    friend std::ostream& operator<<(std::ostream& os, const Instruct& other) {
      os << "Instruct " << other.op_type << " " << other.place();
      return os;
    }
  };

  struct Argument {
    std::string name;
    const Type* type{};
    // Weight is a special kind of argument, it is marked as weight explicitly
    // so that some weight related optimization can take place.
    bool is_weight{false};
  };

  Argument& AsArgument(const std::string& name) {
    auto& x = AsArgument();
    x.name = name;
    return x;
  }

  Instruct& AsInstruct(const std::string& op_type,
                       std::vector<std::unique_ptr<KernelBase>>&& kernels,
                       const std::shared_ptr<OpLite>& op) {
    auto& x = AsInstruct();
    x.op_type = op_type;
    x.op = op;
    x.valid_kernels = std::move(kernels);
    return x;
  }

  // Set roles.
  Argument& AsArgument() {
    if (role_ != Role::kUnk) {
      CHECK(role_ == Role::kArgument);
      return *argument_;
    }
    role_ = Role::kArgument;
    argument_.reset(new Argument);
    return *argument_;
  }
  Instruct& AsInstruct() {
    if (role_ != Role::kUnk) {
      CHECK(role_ == Role::kInstruct);
      return *instruct_;
    }
    role_ = Role::kInstruct;
    instruct_.reset(new Instruct);
    return *instruct_;
  }

  friend std::ostream& operator<<(std::ostream& os, Node& other) {
    os << static_cast<int>(other.role_) << " ";
    if (!other.IsRoleSet()) {
      os << "Unk role node";
    }
    if (other.IsArgument()) {
      auto& arg = other.AsArgument();
      os << "Argument " << arg.name;
    }
    if (other.IsInstruct()) {
      auto& arg = other.AsInstruct();
      os << "Instruct " << arg.op_type;
    }
    return os;
  }

  // Check roles.
  bool IsRoleSet() const { return role_ != Role::kUnk; }
  bool IsInstruct() const { return role_ == Role::kInstruct; }
  bool IsArgument() const { return role_ == Role::kArgument; }

 private:
  // Either instruct_ or argument_ is used.
  std::unique_ptr<Instruct> instruct_;
  std::unique_ptr<Argument> argument_;

  Role role_{Role::kUnk};
};
}  // namespace mir
}  // namespace lite
}  // namespace paddle
