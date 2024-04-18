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

#include "paddle/cinn/operator_fusion/pattern_node.h"
#include "paddle/cinn/operator_fusion/policy/policy_manager.h"
#include "paddle/cinn/operator_fusion/policy/relative_judge_policy.h"
#include "paddle/cinn/operator_fusion/utils.h"

namespace cinn::fusion {

template <typename T>
using PatternNodePtrSet = std::unordered_set<PatternNodePtr<T>>;

template <typename T>
class PatternGraph {
 public:
  PatternGraph(const std::vector<PatternContent<T>>& nodes,
               const std::vector<pir::Value>& outputs,
               const PolicyManager<T> policy_manager,
               const PolicyManager<T> topo_manager);

  std::vector<PatternNodePtr<T>> ClusterOps();

  void SinkTrivialPattern();
  void HorizontalFusion();
  void ReduceLiftReduceTree();
  void ReduceTreeGrown();
  void ReduceTree_Trivial_Fusion();

  void RemoveNode(const PatternNodePtr<T>& node);
  void AppendNode(const PatternNodePtr<T>& node);
  std::string GraphInfo() const;
  PatternNodePtr<T> MergeNode(const PatternNodePtr<T>& upstream,
                              const PatternNodePtr<T>& downstream);
  std::vector<PatternNodePtr<T>> SortByTopoOrder();

 public:
  PatternNodePtrSet<T> all_pattern_nodes_;
  std::vector<pir::Value> outputs_;
  PolicyManager<T> policy_manager_;
  PolicyManager<T> topo_manager_;
};

// PatternGraphFusionOperation := (GraphMatcher, GraphOperation)
// SearchAlgorithm := NodePattern | EdgePattern | GraphMatcher
// GraphOperation := Merge2Node | SplitNode | SplitAllAndMergeDownstream

struct NodePattern {};
struct EdgePattern {};
struct GraphPattern {};     // not implemented.
struct NodePairPattern {};  // not implemented.

template <typename Kind,
          typename Phrase,
          typename GraphMatcher,
          typename GraphOperation>
struct SearchAlgorithm {};

template <typename Phrase, typename GraphMatcher, typename GraphOperation>
struct SearchAlgorithm<NodePattern, Phrase, GraphMatcher, GraphOperation> {
  PatternGraph<Phrase>* graph_;
  PatternNodePtrSet<Phrase> visited_nodes;

  explicit SearchAlgorithm(PatternGraph<Phrase>* graph) {
    VLOG(4) << "Create NodePattern algorithm.";
    graph_ = graph;
  }

  PatternNodePtr<Phrase> FindMatchedNode() {
    for (PatternNodePtr<Phrase> iter_node : graph_->all_pattern_nodes_) {
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
      PatternNodePtr<Phrase> node = FindMatchedNode();
      if (node == nullptr) {
        break;
      }
      GraphOperation()(graph_, node);
    }
  }
};

template <typename Phrase, typename GraphMatcher, typename GraphOperation>
struct SearchAlgorithm<NodePairPattern, Phrase, GraphMatcher, GraphOperation> {
  PatternGraph<Phrase>* graph_;
  std::set<std::pair<PatternNodePtr<Phrase>, PatternNodePtr<Phrase>>>
      visited_node_pair;
  explicit SearchAlgorithm(PatternGraph<Phrase>* graph) {
    VLOG(4) << "Create NodePairPattern algorithm.";
    graph_ = graph;
  }
  std::optional<std::pair<PatternNodePtr<Phrase>, PatternNodePtr<Phrase>>>
  FindMatchedPair() {
    for (PatternNodePtr<Phrase> i : graph_->all_pattern_nodes_) {
      for (PatternNodePtr<Phrase> j : graph_->all_pattern_nodes_) {
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
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph, PatternNodePtr<Phrase> node) {
    CHECK_EQ(node->downstream_.size(), 1);
    auto downstream = node->downstream_.at(0);
    auto merged_node = graph->MergeNode(node, downstream);
    graph->RemoveNode(downstream);
    graph->RemoveNode(node);
    VLOG(4) << "MergeReduceTreeOperation: \nupstream " << node->DebugStr()
            << "\ndownstream " << downstream->DebugStr() << "\nmerged "
            << merged_node->DebugStr();
  }
};

struct MergeReduceTreeAndTrivialOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph, PatternNodePtr<Phrase> node) {
    CHECK_EQ(node->downstream_.size(), 1);
    auto downstream = node->downstream_.at(0);
    auto fake_reduce_iter_idx =
        graph->policy_manager_.GetFakeReduceIterIdx(node, downstream);
    PatternNodePtr<Phrase> merged_node = graph->MergeNode(node, downstream);
    std::get<ReduceTreePlusTrivialPattern<Phrase>>(merged_node->stmt_pattern_)
        .fake_reduce_iter_idx = fake_reduce_iter_idx;
    graph->RemoveNode(downstream);
    graph->RemoveNode(node);
    VLOG(4) << "MergeReduceTreeAndTrivialOperation: \nupstream "
            << node->DebugStr() << "\ndownstream " << downstream->DebugStr()
            << "\nmerged " << merged_node->DebugStr();
  }
};

struct LiftReduceToReduceTreeOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph, PatternNodePtr<Phrase> node) {
    const auto& reduce_pattern = ToReducePattern<Phrase>(node->stmt_pattern_);
    node->stmt_pattern_ = ReduceTreePattern<Phrase>({}, reduce_pattern);
    VLOG(4) << "LiftReduceToReduceTreeOperation: \nnode " << node->DebugStr();
  }
};

struct MergeTrivialPatternOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph,
                  PatternNodePtr<Phrase> upstream) {
    std::vector<PatternNodePtr<Phrase>> fusion_candidate =
        upstream->downstream_;
    upstream->downstream_.clear();
    for (const auto& downstream : fusion_candidate) {
      if (std::holds_alternative<ReducePattern<Phrase>>(
              downstream->stmt_pattern_) ||
          std::holds_alternative<TrivialPattern<Phrase>>(
              downstream->stmt_pattern_)) {
        auto merged_node = graph->MergeNode(upstream, downstream);
        graph->RemoveNode(downstream);
        VLOG(4) << "MergeTrivialPatternOperation: \nupstream "
                << upstream->DebugStr() << "\ndownstream "
                << downstream->DebugStr() << "\nmerged "
                << merged_node->DebugStr();
      } else {
        upstream->downstream_.push_back(downstream);
      }
    }
    if (upstream->downstream_.empty()) {
      graph->RemoveNode(upstream);
    }
  }
};

struct LiftToHorizontalFusionPatternOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph, PatternNodePtr<Phrase> i) {
    i->stmt_pattern_ = HorizontalFusionPattern<Phrase>({i->stmt_pattern_});
  }
};

// Matcher

template <typename StmtPattern>
struct AlwaysTrue {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return true;
  }
};

template <typename StmtPattern>
struct StmtPatternGraphMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return GetPatternName(node->stmt_pattern_) == StmtPattern::name();
  }
};

struct CanFuseRxTMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return (std::holds_alternative<ReduceTreePattern<T>>(node->stmt_pattern_) &&
            !node->downstream_.empty() &&
            std::holds_alternative<TrivialPattern<T>>(
                node->downstream_.at(0)->stmt_pattern_));
  }
};

struct CanFuseReduceTreeMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return StmtPatternGraphMatcher<ReduceTreePattern<T>>()(graph, node) &&
           !node->downstream_.empty() &&
           std::holds_alternative<ReduceTreePattern<T>>(
               node->downstream_.at(0)->stmt_pattern_) &&
           graph.policy_manager_.CanFuse(node, node->downstream_.at(0));
  }
};

struct CanFuseReduceTreeAndTrivialMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return StmtPatternGraphMatcher<ReduceTreePattern<T>>()(graph, node) &&
           !node->downstream_.empty() &&
           std::holds_alternative<TrivialPattern<T>>(
               node->downstream_.at(0)->stmt_pattern_) &&
           graph.policy_manager_.CanFuse(node, node->downstream_.at(0));
  }
};

struct HorizontalFusionConstrain {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph,
                  const PatternNodePtr<T>& first,
                  const PatternNodePtr<T>& second) {
    if (!StmtPatternGraphMatcher<HorizontalFusionPattern<T>>()(graph, first)) {
      return false;
    }
    if (!StmtPatternGraphMatcher<HorizontalFusionPattern<T>>()(graph, second)) {
      return false;
    }
    const auto& first_dim = first->sink_op_->result(0)
                                .type()
                                .template dyn_cast<pir::DenseTensorType>()
                                .dims();
    const auto& second_dim = second->sink_op_->result(0)
                                 .type()
                                 .template dyn_cast<pir::DenseTensorType>()
                                 .dims();
    return graph.topo_manager_.CanFuse(first, second) &&
           first_dim == second_dim;
  }
};

struct HorizontalFusionOperation {
  template <typename T>
  void operator()(PatternGraph<T>* graph,
                  const PatternNodePtr<T>& i,
                  const PatternNodePtr<T>& j) {
    CHECK(GetPatternName(i->stmt_pattern_) ==
          HorizontalFusionPattern<T>::name());
    CHECK(GetPatternName(j->stmt_pattern_) ==
          HorizontalFusionPattern<T>::name());
    graph->MergeNode(i, j);
    graph->RemoveNode(i);
    graph->RemoveNode(j);
  }
};

struct NonSinkNodeMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return !node->downstream_.empty();
  }
};

struct IsOutputNodeMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    bool res = IsAnyFirstInSecond(node->sink_op_->results(), graph.outputs_);
    return res;
  }
};

struct IsNotOutputNodeMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    bool res = !IsOutputNodeMatcher()(graph, node);
    return res;
  }
};

template <int N>
struct DownstreamSmallerThan {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return node->downstream_.size() < N;
  }
};

template <typename A, typename B>
struct And {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return A()(graph, node) && B()(graph, node);
  }
};

template <typename A, typename B>
struct Or {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return A()(graph, node) || B()(graph, node);
  }
};

template <typename A>
struct Not {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return !A()(graph, node);
  }
};

template <typename Kind,
          typename Phrase,
          typename GraphMatcher,
          typename GraphOperation>
void GraphTransformer(PatternGraph<Phrase>* graph) {
  VLOG(4) << "Start GraphTransformer...";
  auto alog =
      SearchAlgorithm<Kind, Phrase, GraphMatcher, GraphOperation>(graph);
  alog();
}

}  // namespace cinn::fusion
