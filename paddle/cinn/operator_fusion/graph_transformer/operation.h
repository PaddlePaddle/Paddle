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
#include "paddle/cinn/operator_fusion/pattern_fuser.h"
#include "paddle/cinn/operator_fusion/pattern_graph.h"

namespace cinn::fusion {

// Operation

struct MergeReduceTreeOperation {
  template <typename Phrase>
  PatternNodePtr<Phrase> operator()(PatternGraph<Phrase>* graph,
                                    PatternNodePtr<Phrase> node) {
    PADDLE_ENFORCE_EQ(
        node->downstream().size(),
        1,
        phi::errors::PreconditionNotMet(
            "The downstream of the ReduceTree node should be 1, but got %d.",
            node->downstream().size()));
    auto downstream = node->downstream().at(0);
    auto merged_node = graph->MergeNode(node, downstream, MergePattern<Phrase>);
    graph->RemoveNode(downstream);
    graph->RemoveNode(node);
    VLOG(4) << "MergeReduceTreeOperation: \nupstream " << node->DebugStr()
            << "\ndownstream " << downstream->DebugStr() << "\nmerged "
            << merged_node->DebugStr();
    return merged_node;
  }
};

struct MergeReduceTreeAndTrivialOperation {
  template <typename Phrase>
  PatternNodePtr<Phrase> operator()(PatternGraph<Phrase>* graph,
                                    PatternNodePtr<Phrase> node) {
    PADDLE_ENFORCE_EQ(
        node->downstream().size(),
        1,
        phi::errors::PreconditionNotMet(
            "The downstream of the ReduceTree node should be 1, but got %d.",
            node->downstream().size()));
    auto downstream = node->downstream().at(0);
    auto fake_reduce_iter_idx = graph->policy_manager()
                                    .template GetPolicy<RelativeJudgePolicy>()
                                    ->GetFakeReduceIterIdx(node, downstream);
    const auto merge_pattern_fn = [&fake_reduce_iter_idx](
                                      const StmtPattern<Phrase>& first,
                                      const StmtPattern<Phrase>& secend) {
      auto rt_pattern = std::get<ReduceTreePlusTrivialPattern<Phrase>>(
          MergePattern<Phrase>(first, secend));
      rt_pattern.fake_reduce_iter_idx = fake_reduce_iter_idx;
      return rt_pattern;
    };
    PatternNodePtr<Phrase> merged_node =
        graph->MergeNode(node, downstream, merge_pattern_fn);
    graph->RemoveNode(downstream);
    graph->RemoveNode(node);
    VLOG(4) << "MergeReduceTreeAndTrivialOperation: \nupstream "
            << node->DebugStr() << "\ndownstream " << downstream->DebugStr()
            << "\nmerged " << merged_node->DebugStr();
    return merged_node;
  }
};

struct LiftReduceToReduceTreeOperation {
  template <typename Phrase>
  PatternNodePtr<Phrase> operator()(PatternGraph<Phrase>* graph,
                                    PatternNodePtr<Phrase> node) {
    const auto& reduce_pattern = ToReducePattern<Phrase>(node->stmt_pattern());
    node->set_stmt_pattern(ReduceTreePattern<Phrase>({}, reduce_pattern));
    return node;
  }
};

struct MergeTrivialPatternOperation {
  template <typename Phrase>
  PatternNodePtr<Phrase> operator()(PatternGraph<Phrase>* graph,
                                    PatternNodePtr<Phrase> node) {
    PADDLE_ENFORCE_EQ(
        node->downstream().size(),
        1,
        phi::errors::PreconditionNotMet("The downstream of the Sink Trivial "
                                        "Pattern node should be 1, but got %d.",
                                        node->downstream().size()));
    const auto& downstream = node->downstream().at(0);
    auto merged_node = graph->MergeNode(node, downstream, MergePattern<Phrase>);
    graph->RemoveNode(downstream);
    graph->RemoveNode(node);
    VLOG(4) << "MergeTrivialPatternOperation: \nupstream " << node->DebugStr()
            << "\ndownstream " << downstream->DebugStr() << "\nmerged "
            << merged_node->DebugStr();
    return merged_node;
  }
};

struct LiftToHorizontalFusionPatternOperation {
  template <typename Phrase>
  PatternNodePtr<Phrase> operator()(PatternGraph<Phrase>* graph,
                                    PatternNodePtr<Phrase> node) {
    node->set_stmt_pattern(HorizontalFusionPattern<Phrase>(
        {typename HorizontalFusionPattern<Phrase>::PaddingStmtPattern(
            node->stmt_pattern(), {})}));
    return node;
  }
};

struct LiftToAnchorPatternOperation {
  template <typename Phrase>
  PatternNodePtr<Phrase> operator()(PatternGraph<Phrase>* graph,
                                    PatternNodePtr<Phrase> node) {
    std::vector<pir::Operation*> ops = GetOpsInPattern(node->stmt_pattern());
    // TODO(@wuzhanfei) move sink_op into pattern (currently, part of pattern
    // type has sink and the others not) then, update logic here
    PADDLE_ENFORCE_EQ(
        node->sink_op()->num_results(),
        1,
        phi::errors::PreconditionNotMet(
            "Op with multi output value can not lift to AnchorPattern"));
    pir::Value anchor = node->sink_op()->result(0);
    node->set_stmt_pattern(AnchorPattern<Phrase>(
        ops,
        anchor,
        AnchorState<Phrase>({InitExprPromise(node->stmt_pattern(), anchor)})));
    return node;
  }
};

struct FuseUpstreamAnchorOperation {
  template <typename Phrase>
  PatternNodePtr<Phrase> operator()(PatternGraph<Phrase>* graph,
                                    const PatternNodePtr<Phrase>& upstream,
                                    const PatternNodePtr<Phrase>& downstream) {
    auto optional_transform_route =
        graph->policy_manager()
            .template GetPolicy<AnchorSearchPolicy>()
            ->FindUpstreamAnchorTransformRoute(upstream, downstream);
    PADDLE_ENFORCE_NE(
        optional_transform_route,
        std::nullopt,
        phi::errors::PreconditionNotMet("Transform Route Not Found"));

    auto transform_route = optional_transform_route.value();

    const auto merge_pattern_fn = [transform_route](
                                      const StmtPattern<Phrase>& source,
                                      const StmtPattern<Phrase>& destination) {
      auto new_anchor_pattern = std::get<AnchorPattern<Phrase>>(
          MergePattern<Phrase>(source, destination));
      auto transformed_anchor_state = ApplyAnchorTransformRoute<Phrase>(
          GetAnchorState(std::get<AnchorPattern<Phrase>>(destination)),
          transform_route);
      new_anchor_pattern.anchor_state.update(
          GetAnchorState(std::get<AnchorPattern<Phrase>>(source)));
      new_anchor_pattern.anchor_state.update(transformed_anchor_state);
      return new_anchor_pattern;
    };

    auto merged_node = graph->MergeNode(upstream, downstream, merge_pattern_fn);
    graph->RemoveNode(upstream);
    graph->RemoveNode(downstream);
    VLOG(4) << "upstream anchor: "
            << std::get<AnchorPattern<Phrase>>(upstream->stmt_pattern())
                   .anchor()
                   .impl()
            << ", downstream anchor: "
            << std::get<AnchorPattern<Phrase>>(downstream->stmt_pattern())
                   .anchor()
                   .impl()
            << ", merged node anchor: "
            << std::get<AnchorPattern<Phrase>>(merged_node->stmt_pattern())
                   .anchor()
                   .impl();
    return merged_node;
  }
};

struct FuseDownstreamAnchorOperation {
  template <typename Phrase>
  PatternNodePtr<Phrase> operator()(PatternGraph<Phrase>* graph,
                                    const PatternNodePtr<Phrase>& upstream,
                                    const PatternNodePtr<Phrase>& downstream) {
    auto optional_transform_route =
        graph->policy_manager()
            .template GetPolicy<AnchorSearchPolicy>()
            ->FindDownstreamAnchorTransformRoute(upstream, downstream);

    PADDLE_ENFORCE_NE(
        optional_transform_route,
        std::nullopt,
        phi::errors::PreconditionNotMet("Transform Route Not Found"));

    auto transform_route = optional_transform_route.value();

    const auto merge_pattern_fn = [transform_route](
                                      const StmtPattern<Phrase>& destination,
                                      const StmtPattern<Phrase>& source) {
      auto new_anchor_pattern = std::get<AnchorPattern<Phrase>>(
          MergePattern<Phrase>(source, destination));
      auto transformed_anchor_state = ApplyAnchorTransformRoute<Phrase>(
          GetAnchorState(std::get<AnchorPattern<Phrase>>(destination)),
          transform_route);
      new_anchor_pattern.anchor_state.update(
          GetAnchorState(std::get<AnchorPattern<Phrase>>(source)));
      new_anchor_pattern.anchor_state.update(transformed_anchor_state);
      return new_anchor_pattern;
    };

    auto merged_node = graph->MergeNode(upstream, downstream, merge_pattern_fn);
    graph->RemoveNode(upstream);
    graph->RemoveNode(downstream);
    VLOG(4) << "upstream anchor: "
            << std::get<AnchorPattern<Phrase>>(upstream->stmt_pattern())
                   .anchor()
                   .impl()
            << ", downstream anchor: "
            << std::get<AnchorPattern<Phrase>>(downstream->stmt_pattern())
                   .anchor()
                   .impl()
            << ", merged node anchor: "
            << std::get<AnchorPattern<Phrase>>(merged_node->stmt_pattern())
                   .anchor()
                   .impl();
    return merged_node;
  }
};

struct SplitRecomputeOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph,
                  PatternNodePtr<Phrase> upstream) {
    std::vector<PatternNodePtr<Phrase>> fusion_candidate =
        upstream->downstream();
    upstream->ClearDownstream();

    upstream->set_stmt_pattern(RecoverAnchorPatternToTrivial(
        std::get<AnchorPattern<Phrase>>(upstream->stmt_pattern())));

    for (const auto& downstream : fusion_candidate) {
      bool can_fuse =
          std::holds_alternative<ReducePattern<Phrase>>(
              downstream->stmt_pattern()) ||
          std::holds_alternative<TrivialPattern<Phrase>>(
              downstream->stmt_pattern()) ||
          std::holds_alternative<ReduceTreePattern<Phrase>>(
              downstream->stmt_pattern()) ||
          std::holds_alternative<ReduceTreePlusTrivialPattern<Phrase>>(
              downstream->stmt_pattern()) ||
          std::holds_alternative<AnchorPattern<Phrase>>(
              downstream->stmt_pattern());

      if (can_fuse) {
        auto merged_node =
            graph->MergeNode(upstream, downstream, MergePattern<Phrase>);
        graph->RemoveNode(downstream);
        VLOG(4) << "Spliting recomputable anchor pattern: \nupstream "
                << upstream->DebugStr() << "\ndownstream "
                << downstream->DebugStr() << "\nmerged "
                << merged_node->DebugStr();
      } else {
        upstream->AddNodeToDownstream(downstream);
      }
    }
    if (upstream->downstream().empty()) {
      graph->RemoveNode(upstream);
    }
  }
};

struct HorizontalFusionOperation {
  template <typename Phrase>
  PatternNodePtr<Phrase> operator()(PatternGraph<Phrase>* graph,
                                    const PatternNodePtr<Phrase>& i,
                                    const PatternNodePtr<Phrase>& j) {
    VLOG(4) << "Start HorizontalFusionOperation";
    PADDLE_ENFORCE_EQ(
        GetPatternName(i->stmt_pattern()),
        HorizontalFusionPattern<Phrase>::name(),
        phi::errors::PreconditionNotMet(
            "The pattern of the first node should be HorizontalFusionPattern, "
            "but got %s.",
            GetPatternName(i->stmt_pattern())));
    PADDLE_ENFORCE_EQ(
        GetPatternName(j->stmt_pattern()),
        HorizontalFusionPattern<Phrase>::name(),
        phi::errors::PreconditionNotMet(
            "The pattern of the second node should be HorizontalFusionPattern, "
            "but got %s.",
            GetPatternName(j->stmt_pattern())));
    auto merged_node = graph->MergeNode(i, j, MergePattern<Phrase>);
    VLOG(4) << "MergeHorizontalPattern: \ni " << i->DebugStr() << "\nj "
            << j->DebugStr() << "\nmerged " << merged_node->DebugStr();
    graph->RemoveNode(i);
    graph->RemoveNode(j);
    return merged_node;
  }
};

}  // namespace cinn::fusion
