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
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return true;
  }
};

template <typename StmtPattern>
struct StmtPatternGraphMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return GetPatternName(node->stmt_pattern()) == StmtPattern::name();
  }
};

struct CanFuseRxTMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return (
        std::holds_alternative<ReduceTreePattern<T>>(node->stmt_pattern()) &&
        !node->downstream().empty() &&
        std::holds_alternative<TrivialPattern<T>>(
            node->downstream().at(0)->stmt_pattern()));
  }
};

struct SinkTrivialMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return StmtPatternGraphMatcher<TrivialPattern<T>>()(graph, node) &&
           node->downstream().size() == 1 &&
           (std::holds_alternative<ReducePattern<Phrase>>(
                node->downstream().at(0)->stmt_pattern()) ||
            std::holds_alternative<TrivialPattern<Phrase>>(
                node->downstream().at(0)->stmt_pattern()));
  }
};

struct CanFuseReduceTreeMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return StmtPatternGraphMatcher<ReduceTreePattern<T>>()(graph, node) &&
           !node->downstream().empty() &&
           std::holds_alternative<ReduceTreePattern<T>>(
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
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return StmtPatternGraphMatcher<ReduceTreePattern<T>>()(graph, node) &&
           !node->downstream().empty() &&
           std::holds_alternative<TrivialPattern<T>>(
               node->downstream().at(0)->stmt_pattern()) &&
           graph.policy_manager()
               .template GetPolicy<GeneralTopoPolicy>()
               ->CanFuse(node, node->downstream().at(0)) &&
           graph.policy_manager()
               .template GetPolicy<RelativeJudgePolicy>()
               ->CanFuse(node, node->downstream().at(0));
  }
};

struct RecomputeNodeMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return StmtPatternGraphMatcher<AnchorPattern<T>>()(graph, node) &&
           node->downstream().size() > 1 &&
           (node->stmt_pattern.can_recompute());
  }
};

struct HasUpstreamAnchorMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph,
                  const PatternNodePtr<T>& upstream,
                  const PatternNodePtr<T>& downstream) {
    return graph.policy_manager()
        .template GetPolicy<AnchorSearchPolicy>()
        ->HasUpstreamAnchor(upstream, downstream);
  }
};

struct HasDownstreamAnchorMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph,
                  const PatternNodePtr<T>& upstream,
                  const PatternNodePtr<T>& downstream) {
    return graph.policy_manager()
        .template GetPolicy<AnchorSearchPolicy>()
        ->HasDownstreamAnchor(upstream, downstream);
  }
};

template <typename T>
struct HorizontalFusionMatcher {
  bool operator()(const PatternGraph<T>& graph,
                  const PatternNodePtr<T>& lhs,
                  const PatternNodePtr<T>& rhs) {
    if (!StmtPatternGraphMatcher<HorizontalFusionPattern<T>>()(graph, lhs)) {
      return false;
    }
    if (!StmtPatternGraphMatcher<HorizontalFusionPattern<T>>()(graph, rhs)) {
      return false;
    }
    const auto& lhs_pattern =
        std::get<HorizontalFusionPattern<T>>(lhs->stmt_pattern());
    const auto& rhs_pattern =
        std::get<HorizontalFusionPattern<T>>(rhs->stmt_pattern());

    return graph.topo_manager().CanFuse(lhs, rhs) &&
           IsLoopFrameworkEqual(lhs_pattern.padding_patterns_.back().pattern,
                                rhs_pattern.padding_patterns_.back().pattern);
  }
};

struct IsOutputNodeMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    bool res = IsAnyFirstInSecond(node->sink_op()->results(), graph.outputs());
    return res;
  }
};

template <int N>
struct DownstreamSmallerThan {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return node->downstream().size() < N;
  }
};

template <int N>
struct DownstreamGreaterThan {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return node->downstream().size() > N;
  }
};

template <typename A, typename B>
struct And {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return A()(graph, node) && B()(graph, node);
  }
};

template <typename... Args>
struct Or {};

template <typename A>
struct Or<A> {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return A()(graph, node);
  }
};

template <typename A, typename... Args>
struct Or<A, Args...> {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return A()(graph, node) || Or<Args...>()(graph, node);
  }
};

template <typename A>
struct Not {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return !A()(graph, node);
  }
};

}  // namespace cinn::fusion
