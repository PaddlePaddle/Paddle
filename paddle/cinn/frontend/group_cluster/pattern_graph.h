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
               const std::vector<pir::Value>& outputs,
               const policy::PolicyManager policy_manager,
               const policy::PolicyManager topo_manager);

  std::vector<PatternNodePtr> ClusterOps(bool with_horizontal_fusion = false);

 private:
  void SinkTrivialPattern();
  void HorizontalFusion();
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
  friend class HorizontalFusionOperation;
  friend class LiftToHorizontalFusionPattern;
  friend class IsNodeOutput;

 public:
  PatternNodePtrSet all_pattern_nodes_;
  PatternNodePtrSet entrance_nodes_;  // bugs here. dont' use this.
  PatternNodePtrSet exit_nodes_;      // bugs here. dont' use this.
  std::vector<pir::Value> outputs_;
  policy::PolicyManager policy_manager_;
  policy::PolicyManager topo_manager_;
};

// PatternGraphFusionOperation := (GraphMatcher, GraphOperation)
// SearchAlgorithm := NodePattern | EdgePattern | GraphMatcher
// GraphOperation := Merge2Node | SplitNode | SplitAllAndMergeDownstream

struct NodePattern {};
struct EdgePattern {};
struct GraphPattern {};     // not implemented.
struct NodePairPattern {};  // not implemented.

template <typename Kind, typename GraphMatcher, typename GraphOperation>
struct SearchAlgorithm {};

template <typename GraphMatcher, typename GraphOperation>
struct SearchAlgorithm<NodePattern, GraphMatcher, GraphOperation> {
  PatternGraph* graph_;
  PatternNodePtrSet visited_nodes;

  explicit SearchAlgorithm(PatternGraph* graph) {
    VLOG(4) << "Create NodePattern algorithm.";
    graph_ = graph;
  }

  PatternNodePtr FindMatchedNode() {
    for (PatternNodePtr iter_node : graph_->all_pattern_nodes_) {
      if (GraphMatcher()(*graph_, iter_node) &&
          !visited_nodes.count(iter_node)) {
        visited_nodes.insert(iter_node);
        VLOG(4) << "Find Matched Node: " << iter_node;
        return iter_node;
      }
    }
    VLOG(4) << "Can't find matched node any more.";
    return nullptr;
  }

  void operator()() {
    while (true) {
      PatternNodePtr node = FindMatchedNode();
      if (node == nullptr) {
        break;
      }
      GraphOperation()(graph_, node);
    }
  }
};

template <typename GraphMatcher, typename GraphOperation>
struct SearchAlgorithm<NodePairPattern, GraphMatcher, GraphOperation> {
  PatternGraph* graph_;
  std::set<std::pair<PatternNodePtr, PatternNodePtr>> visited_node_pair;
  explicit SearchAlgorithm(PatternGraph* graph) {
    VLOG(4) << "Create NodePairPattern algorithm.";
    graph_ = graph;
  }
  std::optional<std::pair<PatternNodePtr, PatternNodePtr>> FindMatchedPair() {
    for (PatternNodePtr i : graph_->all_pattern_nodes_) {
      for (PatternNodePtr j : graph_->all_pattern_nodes_) {
        if (i == j) continue;
        const auto& pair = std::make_pair(i, j);
        if (GraphMatcher()(*graph_, i, j) && !visited_node_pair.count(pair)) {
          visited_node_pair.insert(pair);
          VLOG(4) << "Find Matched Node Pair: (" << i << ", " << j << ")";
          return pair;
        }
      }
    }
    VLOG(4) << "Can't find matched node any more.";
    return {};
  }
  void operator()() {
    while (true) {
      const auto& node = FindMatchedPair();
      if (!node.has_value()) break;
      const auto& [i, j] = node.value();
      GraphOperation()(graph_, i, j);
    }
  }
};

// Operation

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

struct LiftToHorizontalFusionPattern {
  void operator()(PatternGraph* graph, PatternNodePtr i) {
    graph->PrintGraph();
    VLOG(4) << "GetPatternName : " << GetPatternName(i->stmt_pattern_);
    VLOG(4) << "GetOpsInPattern: " << GetOpsInPattern(i->stmt_pattern_).size();
    i->stmt_pattern_ =
        HorizontalFusionPattern(GetOpsInPattern(i->stmt_pattern_));
  }
};

// Matcher

template <typename StmtPattern>
struct AlwaysTrue {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return true;
  }
};

template <typename StmtPattern>
struct StmtPatternGraphMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return GetPatternName(node->stmt_pattern_) == StmtPattern::name();
  }
};

struct CanFuseRxTMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return (node->IsReduceTree() && !node->downstream_.empty() &&
            node->downstream_.at(0)->IsTrivial());
  }
};

struct CanFuseReduceTreeMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return StmtPatternGraphMatcher<ReduceTreePattern>()(graph, node) &&
           !node->downstream_.empty() &&
           node->downstream_.at(0)->IsReduceTree() &&
           graph.policy_manager_.CanFuse(node, node->downstream_.at(0));
  }
};

struct HorizontalFusionConstrain {
  bool operator()(const PatternGraph& graph,
                  const PatternNodePtr& i,
                  const PatternNodePtr& j) {
    if (!StmtPatternGraphMatcher<HorizontalFusionPattern>()(graph, i)) {
      return false;
    }
    if (!StmtPatternGraphMatcher<HorizontalFusionPattern>()(graph, j)) {
      return false;
    }
    const auto& i_dim =
        i->sink_op_->result(0).type().dyn_cast<pir::DenseTensorType>().dims();
    const auto& j_dim =
        j->sink_op_->result(0).type().dyn_cast<pir::DenseTensorType>().dims();
    VLOG(4) << "graph.topo_manager_.CanFuse(i, j) = "
            << graph.topo_manager_.CanFuse(i, j);
    return graph.topo_manager_.CanFuse(i, j) && i_dim == j_dim;
  }
};

struct HorizontalFusionOperation {
  void operator()(PatternGraph* graph,
                  const PatternNodePtr& i,
                  const PatternNodePtr& j) {
    VLOG(4) << "Start HorizontalFusionOperation";
    CHECK(GetPatternName(i->stmt_pattern_) == HorizontalFusionPattern::name());
    CHECK(GetPatternName(j->stmt_pattern_) == HorizontalFusionPattern::name());
    graph->MergeNode(i, j);
    graph->RemoveNode(i);
    graph->RemoveNode(j);
    VLOG(4) << "End HorizontalFusionOperation";
  }
};

struct NonSinkNodeMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return !node->downstream_.empty();
  }
};

struct IsOutputNode {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    bool res = IsAnyFirstInSecond(node->sink_op_->results(), graph.outputs_);
    return res;
  }
};

struct IsNotOutputNode {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    bool res = !IsOutputNode()(graph, node);
    return res;
  }
};

template <int N>
struct DownstreamSmallerThan {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return node->downstream_.size() < N;
  }
};

template <typename A, typename B>
struct And {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return A()(graph, node) && B()(graph, node);
  }
};

template <typename A, typename B>
struct Or {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return A()(graph, node) || B()(graph, node);
  }
};

template <typename A>
struct Not {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return !A()(graph, node);
  }
};

template <typename Kind, typename GraphMatcher, typename GraphOperation>
void GraphTransformer(PatternGraph* graph) {
  VLOG(4) << "Start GraphTransformer...";
  auto alog = SearchAlgorithm<Kind, GraphMatcher, GraphOperation>(graph);
  alog();
}

}  // namespace cinn::frontend::group_cluster
