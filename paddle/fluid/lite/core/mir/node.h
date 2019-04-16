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
    kUnk = -1,
    kArgument,
    kInstruct,
    kNumRoles /*should be last*/
  };

  struct Instruct {
    std::string op_type;
    Place place;
    // The kernel instances this Instruct contains.
    std::vector<std::unique_ptr<KernelBase>> valid_kernels;
  };

  struct Argument {
    std::string name;
    Place place;
  };

  Argument& AsArgument(const std::string& name) {
    auto& x = AsArgument();
    x.name = name;
    return x;
  }

  Instruct& AsInstruct(const std::string& op_type) {
    auto& x = AsInstruct();
    x.op_type = op_type;
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
  // Check roles.
  bool IsRoleSet() const { return role_ == Role::kUnk; }
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
