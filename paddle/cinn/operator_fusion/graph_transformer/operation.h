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

struct MergeTrivialPatternOperation {
  void operator()(PatternGraph* graph, PatternNodePtr upstream) {
    PADDLE_ENFORCE_GE(upstream->downstream().size(),
                      1,
                      ::common::errors::PreconditionNotMet(
                          "The trivial pattern wait for sinking should has "
                          "at least 1 downstream , but got %d.",
                          upstream->downstream().size()));

    std::vector<PatternNodePtr> fusion_candidate = upstream->downstream();
    upstream->ClearDownstream();

    for (const auto& downstream : fusion_candidate) {
      bool can_fuse =
          std::holds_alternative<ReducePattern>(downstream->stmt_pattern()) ||
          std::holds_alternative<TrivialPattern>(downstream->stmt_pattern()) ||
          std::holds_alternative<ReduceTreePattern>(
              downstream->stmt_pattern()) ||
          std::holds_alternative<ReduceTreePlusTrivialPattern>(
              downstream->stmt_pattern()) ||
          std::holds_alternative<ItersPermutationPattern>(
              downstream->stmt_pattern());

      if (can_fuse) {
        auto merged_node = graph->MergeNode(upstream, downstream, MergePattern);
        merged_node->set_fusion_iters(
            graph->iters_fusion_policy()->SingleDownstreamItersFusion(
                upstream, downstream));
        graph->RemoveNode(downstream);
        VLOG(4) << "Spliting trivial pattern: \nupstream "
                << upstream->DebugStr() << "\ndownstream "
                << downstream->DebugStr() << "\nmerged "
                << merged_node->DebugStr();
        merged_node->AppendInstr(std::make_shared<TrivialInlineInstr>(
            upstream->id(), downstream->id(), merged_node->id()));
      } else {
        upstream->AddNodeToDownstream(downstream);
      }
    }
    if (upstream->downstream().empty()) {
      graph->RemoveNode(upstream);
    }
  }
};

struct MergeReduceTreeOperation {
  PatternNodePtr operator()(PatternGraph* graph, PatternNodePtr node) {
    PADDLE_ENFORCE_EQ(
        node->downstream().size(),
        1,
        ::common::errors::PreconditionNotMet(
            "The downstream of the ReduceTree node should be 1, but got %d.",
            node->downstream().size()));
    auto downstream = node->downstream().at(0);
    auto merged_node = graph->MergeNode(node, downstream, MergePattern);
    merged_node->set_fusion_iters(
        graph->iters_fusion_policy()->SingleDownstreamItersFusion(node,
                                                                  downstream));
    graph->RemoveNode(downstream);
    graph->RemoveNode(node);
    VLOG(4) << "MergeReduceTreeOperation: \nupstream " << node->DebugStr()
            << "\ndownstream " << downstream->DebugStr() << "\nmerged "
            << merged_node->DebugStr();
    merged_node->UpdateTracker();
    return merged_node;
  }
};

struct MergeReduceTreeAndTrivialOperation {
  PatternNodePtr operator()(PatternGraph* graph, PatternNodePtr node) {
    PADDLE_ENFORCE_EQ(
        node->downstream().size(),
        1,
        ::common::errors::PreconditionNotMet(
            "The downstream of the ReduceTree node should be 1, but got %d.",
            node->downstream().size()));
    auto downstream = node->downstream().at(0);
    auto fake_reduce_iter_idx = graph->policy_manager()
                                    .template GetPolicy<RelativeJudgePolicy>()
                                    ->GetFakeReduceIterIdx(node, downstream);
    const auto merge_pattern_fn = [&fake_reduce_iter_idx](
                                      const StmtPattern& first,
                                      const StmtPattern& secend) {
      auto rt_pattern =
          std::get<ReduceTreePlusTrivialPattern>(MergePattern(first, secend));
      rt_pattern.fake_reduce_iter_idx = fake_reduce_iter_idx;
      return rt_pattern;
    };
    PatternNodePtr merged_node =
        graph->MergeNode(node, downstream, merge_pattern_fn);
    merged_node->set_fusion_iters(
        graph->iters_fusion_policy()->SingleDownstreamItersFusion(node,
                                                                  downstream));
    graph->RemoveNode(downstream);
    graph->RemoveNode(node);
    VLOG(4) << "MergeReduceTreeAndTrivialOperation: \nupstream "
            << node->DebugStr() << "\ndownstream " << downstream->DebugStr()
            << "\nmerged " << merged_node->DebugStr();
    merged_node->UpdateTracker();
    return merged_node;
  }
};

struct LiftReduceToReduceTreeOperation {
  PatternNodePtr operator()(PatternGraph* graph, PatternNodePtr node) {
    auto origin_name = node->id();
    const auto& reduce_pattern = std::get<ReducePattern>(node->stmt_pattern());
    node->set_stmt_pattern(ReduceTreePattern(
        {},
        reduce_pattern,
        std::make_shared<FusionTracker>(reduce_pattern.tracker_)));
    VLOG(4) << "Make CopyInstr: " << origin_name << " -> " << node->id();
    node->AppendInstr(std::make_shared<CopyInstr>(origin_name, node->id()));
    return node;
  }
};

struct LiftToHorizontalFusionPatternOperation {
  PatternNodePtr operator()(PatternGraph* graph, PatternNodePtr node) {
    auto origin_name = node->id();
    node->set_stmt_pattern(HorizontalFusionPattern(
        {typename HorizontalFusionPattern::PaddingStmtPattern(
            node->stmt_pattern(), {})},
        std::make_shared<FusionTracker>(
            GetFusionTracker(node->stmt_pattern()))));
    VLOG(4) << "Make CopyInstr: " << origin_name << " -> " << node->id();
    node->AppendInstr(std::make_shared<CopyInstr>(origin_name, node->id()));
    return node;
  }
};

struct LiftToItersPermutationPatternOperation {
  PatternNodePtr operator()(PatternGraph* graph, PatternNodePtr node) {
    PADDLE_ENFORCE_EQ(node->sink_op()->num_results(),
                      1,
                      ::common::errors::PreconditionNotMet(
                          "Op with multi output value can not lift to "
                          "ItersPermutationPattern"));
    std::string origin_name = node->id();
    node->set_stmt_pattern(ItersPermutationPattern(
        GetOpsInPattern(node->stmt_pattern()),
        std::make_shared<FusionTracker>(GetFusionTracker(node->stmt_pattern())),
        graph->iters_fusion_policy()->GetLoopDims(node->fusion_iters())));
    node->AppendInstr(std::make_shared<CopyInstr>(origin_name, node->id()));
    VLOG(4) << "Make CopyInstr: " << origin_name << " -> " << node->id();
    return node;
  }
};

struct FuseItersPermutatioOperation {
  PatternNodePtr operator()(PatternGraph* graph,
                            const PatternNodePtr& upstream,
                            const PatternNodePtr& downstream) {
    VLOG(4) << "Start FuseItersPermutatioOperation";
    VLOG(4) << "Upstream: \n" << upstream->DebugStr();
    VLOG(4) << "Downstream: \n" << downstream->DebugStr();
    const auto rise_transform_route =
        graph->iters_fusion_policy()->GetItersTransformRoute(downstream,
                                                             upstream);
    const auto sink_transform_route =
        graph->iters_fusion_policy()->GetItersTransformRoute(upstream,
                                                             downstream);
    PADDLE_ENFORCE_EQ(
        rise_transform_route != std::nullopt ||
            sink_transform_route != std::nullopt,
        true,
        ::common::errors::NotFound("Can not find Transform route."));
    const bool is_rise = rise_transform_route != std::nullopt;
    const auto transform_route =
        is_rise ? rise_transform_route.value() : sink_transform_route.value();

    const auto merge_pattern_fn =
        [=](const StmtPattern& upstream,
            const StmtPattern& downstream) -> StmtPattern {
      const auto upstream_pattern = std::get<ItersPermutationPattern>(upstream);
      const auto downstream_pattern =
          std::get<ItersPermutationPattern>(downstream);
      return ItersPermutationPattern(
          UniqueConcatVector(GetOpsInPattern(upstream),
                             GetOpsInPattern(downstream)),
          std::make_shared<FusionTracker>(upstream_pattern.tracker_,
                                          downstream_pattern.tracker_),
          is_rise ? upstream_pattern.loop_dims_
                  : downstream_pattern.loop_dims_);
    };
    auto merged_node = graph->MergeNode(upstream, downstream, merge_pattern_fn);
    merged_node->set_fusion_iters(
        graph->iters_fusion_policy()->MultiDownstreamItersFusion(
            upstream,
            downstream,
            is_rise
                ? FusionItersManager::FusionDirection::downstream2upstream
                : FusionItersManager::FusionDirection::upstream2downstream));
    const auto update_tracker_fn = [=](const PatternNodePtr& source,
                                       const PatternNodePtr& target) {
      const std::string source_tmp_id = GetNewTmpId(source->id());
      merged_node->AppendInstr(std::make_shared<ItersTransformInstr>(
          source->id(), target->id(), source_tmp_id, transform_route));
      const std::vector<std::string> names =
          is_rise ? std::vector<std::string>({target->id(), source_tmp_id})
                  : std::vector<std::string>({source_tmp_id, target->id()});
      merged_node->AppendInstr(
          std::make_shared<CombineInstr>(names, merged_node->id()));
    };
    if (is_rise) {
      update_tracker_fn(downstream, upstream);
    } else {
      update_tracker_fn(upstream, downstream);
    }
    graph->RemoveNode(upstream);
    graph->RemoveNode(downstream);
    VLOG(4) << "Merged: \n" << merged_node->DebugStr();
    return merged_node;
  }
};

struct SplitRecomputeOperation {
  void operator()(PatternGraph* graph, PatternNodePtr upstream) {
    auto origin_name = upstream->id();
    VLOG(4) << "SplitRecomputeOperation: upstream tracker is: "
            << GetFusionTracker(upstream->stmt_pattern())->DebugStr();

    const auto trivial_pattern = TrivialPattern(
        GetOpsInPattern(upstream->stmt_pattern()),
        upstream->sink_op(),
        std::make_shared<FusionTracker>(
            std::get<ItersPermutationPattern>(upstream->stmt_pattern())
                .tracker_));
    upstream->set_stmt_pattern(trivial_pattern);
    VLOG(4) << "Make CopyInstr: " << origin_name << " -> " << upstream->id();
    upstream->AppendInstr(
        std::make_shared<CopyInstr>(origin_name, upstream->id()));
    VLOG(4) << "After SplitRecomputeOperation: upstream tracker is: "
            << GetFusionTracker(upstream->stmt_pattern())->DebugStr();
    MergeTrivialPatternOperation()(graph, upstream);
  }
};

struct HorizontalFusionOperation {
  PatternNodePtr operator()(PatternGraph* graph,
                            const PatternNodePtr& i,
                            const PatternNodePtr& j) {
    VLOG(4) << "Start HorizontalFusionOperation";
    PADDLE_ENFORCE_EQ(
        GetPatternName(i->stmt_pattern()),
        HorizontalFusionPattern::name(),
        ::common::errors::PreconditionNotMet(
            "The pattern of the first node should be HorizontalFusionPattern, "
            "but got %s.",
            GetPatternName(i->stmt_pattern())));
    PADDLE_ENFORCE_EQ(
        GetPatternName(j->stmt_pattern()),
        HorizontalFusionPattern::name(),
        ::common::errors::PreconditionNotMet(
            "The pattern of the second node should be HorizontalFusionPattern, "
            "but got %s.",
            GetPatternName(j->stmt_pattern())));
    auto merged_node = graph->MergeNode(i, j, MergePattern);
    VLOG(4) << "MergeHorizontalPattern: \ni " << i->DebugStr() << "\nj "
            << j->DebugStr() << "\nmerged " << merged_node->DebugStr();
    graph->RemoveNode(i);
    graph->RemoveNode(j);
    merged_node->UpdateTracker();
    return merged_node;
  }
};

}  // namespace cinn::fusion
