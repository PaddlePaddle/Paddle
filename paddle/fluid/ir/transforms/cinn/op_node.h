// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <memory>
#include "paddle/fluid/ir/transforms/fusion_merge_util.h"
#include "paddle/ir/core/operation.h"

namespace ir {
namespace api {

class OpNode {
 public:
  explicit OpNode(const ::ir::Operation* node) : node_(node) {}

  OpPatternKind kind() const {
    auto kind = GetOpKind(node_->name());
    if (kind == kBroadcast) {
      // As binary op was defined as broadcast, actually it should be
      // element-wise.
      if (node_->name() != "broadcast_to") {
        return kElementWise;
      }
    }
    return kind;
  }

  bool operator==(const OpNode& other) const { return node_ == other.node_; }

  bool operator<(const OpNode& other) const { return node_ < other.node_; }

  const Operation* Op() const { return node_; }

 private:
  friend struct std::hash<OpNode>;

  const Operation* node_;
};

}  // namespace api
}  // namespace ir

namespace std {

template <>
struct hash<ir::api::OpNode> {
  size_t operator()(const ir::api::OpNode& obj) const {
    return std::hash<size_t>()(reinterpret_cast<size_t>(obj.node_));
  }
};

}  // namespace std
