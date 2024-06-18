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
#include "paddle/cinn/operator_fusion/pattern_graph.h"

namespace cinn::fusion {
// Matcher

struct AlwaysTrue {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return true;
  }
};

template <typename StmtPattern>
struct StmtPatternGraphMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return GetPatternName(node->stmt_pattern()) == StmtPattern::name();
  }
};

struct CanFuseRxTMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return (std::holds_alternative<ReduceTreePattern>(node->stmt_pattern()) &&
            !node->downstream().empty() &&
            std::holds_alternative<TrivialPattern>(
                node->downstream().at(0)->stmt_pattern()));
  }
};

struct CanFuseReduceTreeMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return StmtPatternGraphMatcher<ReduceTreePattern>()(graph, node) &&
           !node->downstream().empty() &&
           std::holds_alternative<ReduceTreePattern>(
               node->downstream().at(0)->stmt_pattern()) &&
           graph.policy_manager()
               .template GetPolicy<GeneralTopoPolicy>()
               ->CanFuse(node, node->downstream().at(0)) &&
           graph.policy_manager()
               .template GetPolicy<RelativeJudgePolicy>()
               ->CanFuse(node, node->downstream().at(0));
  }
};

struct CanFuseReduceTreeAndTrivialMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return StmtPatternGraphMatcher<ReduceTreePattern>()(graph, node) &&
           !node->downstream().empty() &&
           std::holds_alternative<TrivialPattern>(
               node->downstream().at(0)->stmt_pattern()) &&
           graph.policy_manager()
               .template GetPolicy<GeneralTopoPolicy>()
               ->CanFuse(node, node->downstream().at(0)) &&
           graph.policy_manager()
               .template GetPolicy<RelativeJudgePolicy>()
               ->CanFuse(node, node->downstream().at(0));
  }
};

struct LiftToAnchorPatternMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    bool not_reduce_tree =
        !StmtPatternGraphMatcher<ReduceTreePattern>()(graph, node) &&
        !StmtPatternGraphMatcher<ReduceTreePlusTrivialPattern>()(graph, node);
    bool reduce_tree_with_single_reduce =
        StmtPatternGraphMatcher<ReduceTreePattern>()(graph, node) &&
        std::get<ReduceTreePattern>(node->stmt_pattern()).childs().size() == 0;
    return not_reduce_tree || reduce_tree_with_single_reduce;
  }
};

struct RecomputeNodeMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return StmtPatternGraphMatcher<AnchorPattern>()(graph, node) &&
           node->downstream().size() >= 1 &&
           (std::get<AnchorPattern>(node->stmt_pattern()).can_recompute());
  }
};

struct HasUpstreamAnchorMatcher {
  bool operator()(const PatternGraph& graph,
                  const PatternNodePtr& upstream,
                  const PatternNodePtr& downstream) {
    if (!StmtPatternGraphMatcher<AnchorPattern>()(graph, upstream) ||
        !StmtPatternGraphMatcher<AnchorPattern>()(graph, downstream)) {
      return false;
    }
    return graph.policy_manager()
               .template GetPolicy<GeneralTopoPolicy>()
               ->CanFuse(upstream, downstream) &&
           graph.policy_manager()
               .template GetPolicy<AnchorSearchPolicy>()
               ->HasUpstreamAnchor(upstream, downstream);
  }
};

struct HasDownstreamAnchorMatcher {
  bool operator()(const PatternGraph& graph,
                  const PatternNodePtr& upstream,
                  const PatternNodePtr& downstream) {
    if (!StmtPatternGraphMatcher<AnchorPattern>()(graph, upstream) ||
        !StmtPatternGraphMatcher<AnchorPattern>()(graph, downstream)) {
      return false;
    }
    return graph.policy_manager()
               .template GetPolicy<GeneralTopoPolicy>()
               ->CanFuse(upstream, downstream) &&
           graph.policy_manager()
               .template GetPolicy<AnchorSearchPolicy>()
               ->HasDownstreamAnchor(upstream, downstream);
  }
};

struct HorizontalFusionMatcher {
  bool operator()(const PatternGraph& graph,
                  const PatternNodePtr& lhs,
                  const PatternNodePtr& rhs) {
    if (!StmtPatternGraphMatcher<HorizontalFusionPattern>()(graph, lhs)) {
      return false;
    }
    if (!StmtPatternGraphMatcher<HorizontalFusionPattern>()(graph, rhs)) {
      return false;
    }
    const auto& lhs_pattern =
        std::get<HorizontalFusionPattern>(lhs->stmt_pattern());
    const auto& rhs_pattern =
        std::get<HorizontalFusionPattern>(rhs->stmt_pattern());

    return graph.policy_manager()
               .template GetPolicy<GeneralTopoPolicy>()
               ->CanFuse(lhs, rhs) &&
           IsLoopFrameworkEqual(lhs_pattern.padding_patterns_.back().pattern,
                                rhs_pattern.padding_patterns_.back().pattern);
  }
};

struct NonSinkNodeMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return !node->downstream().empty();
  }
};

struct IsOutputNodeMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    bool res = IsAnyFirstInSecond(node->sink_op()->results(), graph.outputs());
    return res;
  }
};

template <int N>
struct DownstreamSmallerThan {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return node->downstream().size() < N;
  }
};

template <int N>
struct DownstreamGreaterThan {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return node->downstream().size() > N;
  }
};

template <typename... Args>
struct And {};

template <typename A>
struct And<A> {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return A()(graph, node);
  }
};

template <typename A, typename... Args>
struct And<A, Args...> {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return A()(graph, node) && And<Args...>()(graph, node);
  }
};

template <typename... Args>
struct Or {};

template <typename A>
struct Or<A> {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return A()(graph, node);
  }
};

template <typename A, typename... Args>
struct Or<A, Args...> {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return A()(graph, node) || Or<Args...>()(graph, node);
  }
};

template <typename A>
struct Not {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return !A()(graph, node);
  }
};

}  // namespace cinn::fusion
