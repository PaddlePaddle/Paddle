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
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/core/mir/node.h"
#include "paddle/fluid/lite/core/mir/pattern_matcher.h"
#include "paddle/fluid/lite/core/mir/ssa_graph.h"

namespace paddle {
namespace lite {
namespace mir {

class FuseBase {
 public:
  using key2nodes_t = std::map<std::string, Node*>;

  virtual ~FuseBase() = default;

  void operator()(SSAGraph* graph) {
    BuildPattern();
    PerformPatternMatcher(graph);

    for (const auto& matched : key2nodes_) {
      InsertNewNode(graph, matched);
    }

    DeleteInterNodes(graph);
  }

  // Build a PMPattern using PMNode.
  virtual void BuildPattern() = 0;

  // Generate an operator desc with a matched subgraph.
  virtual cpp::OpDesc GenOpDesc(const key2nodes_t& matched) {
    return cpp::OpDesc();
  }

  PMNode* OpNode(const std::string& key) {
    return GetOrCreateNode(key)->assert_is_op();
  }

  PMNode* OpNode(const std::string& key, const std::string& op_type);

  PMNode* VarNode(const std::string& key);

 protected:
  virtual void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) = 0;

 private:
  void PerformPatternMatcher(SSAGraph* graph);

  // Delete nodes that are marked as Intermediate
  void DeleteInterNodes(SSAGraph* graph);

  PMNode* GetOrCreateNode(const std::string& key);

 protected:
  PatternMatcher matcher_;
  std::map<std::string, PMNode*> nodes_;
  std::vector<key2nodes_t> key2nodes_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
