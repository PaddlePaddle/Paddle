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

struct CanFuseReduceTreeMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return StmtPatternGraphMatcher<ReduceTreePattern<T>>()(graph, node) &&
           !node->downstream().empty() &&
           std::holds_alternative<ReduceTreePattern<T>>(
               node->downstream().at(0)->stmt_pattern()) &&
           graph.policy_manager().GetPolicy<GeneralTopoPolicy>().CanFuse(
               node, node->downstream().at(0)) &&
           graph.policy_manager().GetPolicy<RelativeJudgePolicy>().CanFuse(
               node, node->downstream().at(0));
  }
};

struct CanFuseReduceTreeAndTrivialMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return StmtPatternGraphMatcher<ReduceTreePattern<T>>()(graph, node) &&
           !node->downstream().empty() &&
           std::holds_alternative<TrivialPattern<T>>(
               node->downstream().at(0)->stmt_pattern()) &&
           graph.policy_manager().GetPolicy<GeneralTopoPolicy>().CanFuse(
               node, node->downstream().at(0)) &&
           graph.policy_manager().GetPolicy<RelativeJudgePolicy>().CanFuse(
               node, node->downstream().at(0));
  }
};

struct RecomputeNodeMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    // TODO(@wuzhanfei)
    return false;
  }
};

struct HasUpstreamAnchorMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    // TODO(@wuzhanfei)
    return false;
  }
};

struct HasDownstreamAnchorMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    // TODO(@wuzhanfei)
    return false
  }
};

struct HorizontalFusionMatcher {
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
    const auto& first_dim = first->sink_op()
                                ->result(0)
                                .type()
                                .template dyn_cast<pir::DenseTensorType>()
                                .dims();
    const auto& second_dim = second->sink_op()
                                 ->result(0)
                                 .type()
                                 .template dyn_cast<pir::DenseTensorType>()
                                 .dims();
    return graph.policy_manager.GetPolicy<GeneralTopoPolicy>().CanFuse(
               first, second) &&
           first_dim == second_dim;
  }
};

struct NonSinkNodeMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return !node->downstream().empty();
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

}  // namespace cinn::fusion
