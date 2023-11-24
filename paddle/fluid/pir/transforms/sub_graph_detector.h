// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <queue>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/pir/core/builder.h"

namespace pir {

struct SubGraph;
using SubGraphPtr = std::shared_ptr<SubGraph>;
using GroupOpsVec = std::vector<pir::Operation*>;

class SubgraphDetector {
 public:
  // Tell whether a node is inside a sub-graph.
  using OpClassifier = std::function<bool(pir::Operation*)>;

  SubgraphDetector(pir::Block* block, const OpClassifier& classifier);

  std::vector<GroupOpsVec> operator()();

 protected:
  // Do Op Fusion
  void DoOpFusion();

  void BuildSubGraph();

  // SubGraph Fusion
  void DoSubGraphFusion();

  bool FuseSubGraph(SubGraphPtr subgraph_ptr);
  // check exist depency.
  bool IsDependency(const SubGraphPtr& producer_g,
                    const SubGraphPtr& consumer,
                    const std::unordered_set<SubGraphPtr>& consumers);

  bool IsDependencySimplify(const SubGraphPtr& producer_g,
                            const SubGraphPtr& consumer,
                            const std::unordered_set<SubGraphPtr>& consumers);

 private:
  pir::Block* block_;
  OpClassifier op_classifier_;

  std::vector<pir::Operation*> sort_ops_;
  std::unordered_map<pir::Operation*, size_t> op2id_;
  std::vector<SubGraphPtr> subgraph_list_;
  std::unordered_map<pir::Operation*, SubGraphPtr> subgraph_map_;
};

void ReplaceWithGroupOp(pir::Block* block, const GroupOpsVec& group_ops);

}  // namespace pir
