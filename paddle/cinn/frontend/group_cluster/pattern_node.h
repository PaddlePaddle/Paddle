// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/frontend/group_cluster/common_utils.h"

namespace cinn::frontend::group_cluster {

struct PatternNode {
  using PatternNodePtr = std::shared_ptr<PatternNode>;

  explicit PatternNode(pir::Operation* op);
  explicit PatternNode(PatternNodePtr fused_up_node,
                       PatternNodePtr fused_down_node);

  bool IsTrivial() const;
  bool IsReduce() const;
  bool IsReduceTree() const;
  bool IsUnsupport() const;
  bool IsReduceTrivial() const;

  std::vector<pir::Operation*> GetOps() const;

  StmtPattern stmt_pattern_;
  pir::Operation* sink_op_;

  std::vector<PatternNodePtr> upstream_;
  std::vector<PatternNodePtr> downstream_;

  std::string DebugStr() const;
};

using PatternNodePtr = PatternNode::PatternNodePtr;
}  // namespace cinn::frontend::group_cluster
