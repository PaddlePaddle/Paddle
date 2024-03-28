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

#include "paddle/cinn/frontend/group_cluster/cluster_policy/policy_manager.h"
#include "paddle/cinn/frontend/group_cluster/cluster_policy/relative_judge_policy.h"
#include "paddle/cinn/frontend/group_cluster/common_utils.h"
#include "paddle/cinn/frontend/group_cluster/pattern_node.h"

namespace cinn::frontend::group_cluster {

struct PatternNodePtrHash {
  size_t operator()(const PatternNodePtr& node) const {
    return std::hash<PatternNode*>()(node.get());
  }
};

struct PatternNodePtrCompare {
  bool operator()(const std::shared_ptr<PatternNode>& a,
                  const std::shared_ptr<PatternNode>& b) const {
    return a.get() == b.get();
  }
};

using PatternNodePtrSet = std::
    unordered_set<PatternNodePtr, PatternNodePtrHash, PatternNodePtrCompare>;

class PatternGraph {
 public:
  PatternGraph(const std::vector<pir::Operation*>& ops,
               const policy::PolicyManager policy_manager);

  std::vector<PatternNodePtr> ClusterOps();

 private:
  void SinkTrivialPattern();
  void FuseReducePattern();
  void ReduceLiftReduceTree();
  void ReduceTreeGrown();
  void ReduceTree_Trivial_Fusion();

  void RemoveNode(const PatternNodePtr& node);
  void AppendNode(const PatternNodePtr& node);
  void PrintGraph();
  PatternNodePtr MergeNode(const PatternNodePtr& upstream,
                           const PatternNodePtr& downstream);
  std::vector<PatternNodePtr> SortByTopoOrder();

  friend class TrivialPatternMerge;
  friend class LiftReduceToReduceTree;
  friend class CanFuseReduceTreeMatcher;
  friend class MergeReduceTreeOperation;
  friend class FuseReduceTreeAndTrivial;

 public:
  PatternNodePtrSet all_pattern_nodes_;
  PatternNodePtrSet entrance_nodes_;
  PatternNodePtrSet exit_nodes_;
  const policy::PolicyManager policy_manager_;
};

// PatternGraphFusionOperation := (GraphMatcher, GraphOperation)
// SearchAlorithm := NodePattern | EdgePattern | GraphMatcher
// GraphOperation := Merge2Node | SplitNode | SplitAllAndMergeDownstream

struct NodePattern {};
struct EdgePattern {};
struct GraphPattern {};  // not implemented.
using PatternKind = std::variant<NodePattern, EdgePattern, GraphPattern>;

template <typename GraphMatcher, typename GraphOperation>
struct SearchAlorithm {
  PatternGraph* graph_;
  PatternNodePtrSet visited_nodes;
  SearchAlorithm(PatternGraph* graph) { graph_ = graph; }

  PatternNodePtr FindMatchedNode() {
    for (PatternNodePtr iter_node : graph_->all_pattern_nodes_) {
      if (GraphMatcher()(*graph_, iter_node) &&
          !visited_nodes.count(iter_node)) {
        VLOG(4) << "Find Matched Node: " << iter_node;
        return iter_node;
      }
    }
    VLOG(4) << "Can't find matched node any more.";
    return nullptr;
  }

  void operator()(const NodePattern& p) {
    while (true) {
      PatternNodePtr node = FindMatchedNode();
      if (node == nullptr) {
        break;
      }
      visited_nodes.insert(node);
      GraphOperation()(graph_, node);
    }
  }

  void operator()(const EdgePattern& p) { CHECK(false) << "Not implemented."; }

  void operator()(const GraphPattern& p) { CHECK(false) << "Not implemented."; }
};

// Operation
//
struct MergeReduceTreeOperation {
  void operator()(PatternGraph* graph, PatternNodePtr node) {
    CHECK_EQ(node->downstream_.size(), 1);
    auto downstream = node->downstream_.at(0);
    graph->PrintGraph();
    VLOG(4) << "Start Merge.";
    graph->MergeNode(node, downstream);
    graph->RemoveNode(downstream);
    graph->RemoveNode(node);
    VLOG(4) << "End Graph is: ";
    graph->PrintGraph();
  }
};

struct FuseReduceTreeAndTrivial {
  void operator()(PatternGraph* graph, PatternNodePtr node) {
    CHECK_EQ(node->downstream_.size(), 1);
    auto downstream = node->downstream_.at(0);
    if (graph->policy_manager_.CanFuse(node, downstream)) {
      PatternNodePtr new_node = std::make_shared<PatternNode>(node, downstream);
      graph->AppendNode(new_node);
      graph->RemoveNode(downstream);
      graph->RemoveNode(node);
    }
  }
};

struct LiftReduceToReduceTree {
  void operator()(PatternGraph* graph, PatternNodePtr node) {
    const auto& reduce_pattern = ToReducePattern(node->stmt_pattern_);
    node->stmt_pattern_ = ReduceTreePattern({reduce_pattern}, reduce_pattern);
  }
};

struct TrivialPatternMerge {
  void operator()(PatternGraph* graph, PatternNodePtr upstream) {
    VLOG(4) << "Start Finding Can Merge Trivial Node.";
    VLOG(4) << "Remain pattern node is: " << graph->all_pattern_nodes_.size();
    graph->PrintGraph();
    std::vector<PatternNodePtr> fusion_candidate = upstream->downstream_;
    upstream->downstream_.clear();
    for (const auto& downstream : fusion_candidate) {
      if (downstream->IsReduce() || downstream->IsTrivial()) {
        graph->MergeNode(upstream, downstream);
        graph->RemoveNode(downstream);
      } else {
        upstream->downstream_.push_back(downstream);
      }
    }
    if (upstream->downstream_.empty()) {
      graph->RemoveNode(upstream);
    }
  }
};

template <typename StmtPattern>
struct StmtPatternGraphMatcher {
  PatternKind type() { return NodePattern(); }
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return GetPatternName(node->stmt_pattern_) == StmtPattern::name();
  }
};

struct CanFuseRxTMatcher {
  PatternKind type() { return NodePattern(); }
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return (node->IsReduceTree() && !node->downstream_.empty() &&
            node->downstream_.at(0)->IsTrivial());
  }
};

struct CanFuseReduceTreeMatcher {
  PatternKind type() { return NodePattern(); }
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return StmtPatternGraphMatcher<ReduceTreePattern>()(graph, node) &&
           !node->downstream_.empty() &&
           node->downstream_.at(0)->IsReduceTree() &&
           graph.policy_manager_.CanFuse(node, node->downstream_.at(0));
  }
};

struct NonSinkNodeMatcher {
  PatternKind type() { return NodePattern(); }
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return !node->downstream_.empty();
  }
};

template <int N>
struct DownstreamSmallerThan {
  PatternKind type() { return NodePattern(); }
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return node->downstream_.size() < N;
  }
};

template <typename A, typename B>
struct And {
  PatternKind type() { return A().type(); }
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return A()(graph, node) && B()(graph, node);
  }
};

template <typename A, typename B>
struct Or {
  PatternKind type() { return A().type(); }
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return A()(graph, node) || B()(graph, node);
  }
};

template <typename A>
struct Not {
  PatternKind type() { return A().type(); }
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return !A()(graph, node);
  }
};

template <typename GraphMatcher, typename GraphOperation>
void GraphTransformer(PatternGraph* graph) {
  const auto& pattern_type = GraphMatcher().type();
  std::visit(SearchAlorithm<GraphMatcher, GraphOperation>(graph), pattern_type);
}

}  // namespace cinn::frontend::group_cluster
