// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_TESTING
#include <gtest/gtest_prod.h>
#endif

#include <numeric>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"

namespace paddle {
namespace framework {
namespace ir {

// Some basic torminolygies:
//   - Pattern: a pattern defined as a graph.

// Pattern detector node. This node helps to build a pattern.
struct PDNode {
  // tell whether an ir::Node* is a candidation for a PDNode.
  using teller_t = std::function<bool(Node*)>;

  PDNode(teller_t&& teller, const std::string& name = "")
      : teller_(teller), name_(name) {
    PADDLE_ENFORCE(teller_ != nullptr, "invalid teller functer is set.");
  }

  PDNode(PDNode&& other) = default;

  std::vector<PDNode*> inlinks;
  std::vector<PDNode*> outlinks;

  bool Tell(Node* node) const {
    PADDLE_ENFORCE(teller_ != nullptr, "teller should be set for a PDNode");
    return teller_(node);
  }

  const std::string& name() const { return name_; }

  PDNode(const PDNode&) = delete;
  PDNode& operator=(const PDNode&) = delete;

 private:
  teller_t teller_;
  std::string name_;
};

class PDPattern {
 public:
  using edge_t = std::pair<PDNode*, PDNode*>;

  void AddEdge(PDNode* a, PDNode* b);

  PDNode* NewNode(PDNode::teller_t&& teller, const std::string& name = "");

  const std::vector<std::unique_ptr<PDNode>>& nodes() const { return nodes_; }
  const std::vector<edge_t>& edges() const { return edges_; }

 private:
#ifdef PADDLE_WITH_TESTING
  FRIEND_TEST(PDPattern, AddEdge);
  FRIEND_TEST(PDPattern, NewNode);
#endif

  std::vector<std::unique_ptr<PDNode>> nodes_;
  std::vector<edge_t> edges_;
};

/*
 * GraphPatternDetecter helps to detect the specific patterns in the graph.
 * Input a pattern, output a list of the matched subgraphs/nodes.
 * This helper can be used to support fuse(conv+batchnorm => batchnorm e.g.).
 */
class GraphPatternDetecter {
 public:
  using hit_rcd_t =
      std::pair<Node* /*node in graph*/, PDNode* /*node in pattern*/>;
  using subgraph_t = std::unordered_map<PDNode*, Node*>;

  // Operate on the detected pattern.
  using handle_t =
      std::function<void(const subgraph_t& /*hitted pattern*/, Graph*)>;

  void operator()(Graph* graph, handle_t handler);

  const PDPattern& pattern() const { return pattern_; }
  PDPattern* mutable_pattern() { return &pattern_; }

 private:
  // Mark the nodes that fits the pattern.
  bool MarkPDNodesInGraph(const ir::Graph& graph);

  // Detect all the pattern and output the hit records.
  std::vector<subgraph_t> DetectPatterns();

  // Remove duplicate patterns
  void UniquePatterns(std::vector<subgraph_t>* subgraphs);

#ifdef PADDLE_WITH_TESTING
  FRIEND_TEST(GraphPatternDetecter, MarkPDNodesInGraph);
  FRIEND_TEST(GraphPatternDetecter, DetectPatterns);
#endif

 private:
  PDPattern pattern_;
  std::vector<hit_rcd_t> marked_records_;
  std::unordered_map<const PDNode*, std::unordered_set<Node*>> pdnodes2nodes_;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
