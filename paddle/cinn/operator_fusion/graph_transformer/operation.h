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
    VLOG(4) << "Sink trivial pattern: \nupstream: " << upstream->DebugStr();
    std::vector<PatternNodePtr> fusion_candidate = upstream->downstream();
    upstream->ClearDownstream();

    for (int i = 0; i < fusion_candidate.size(); ++i) {
      const auto& downstream = fusion_candidate[i];
      bool can_fuse =
          std::holds_alternative<ReducePattern>(downstream->stmt_pattern()) ||
          std::holds_alternative<TrivialPattern>(downstream->stmt_pattern()) ||
          std::holds_alternative<ReduceTreePattern>(
              downstream->stmt_pattern()) ||
          std::holds_alternative<ReduceTreePlusTrivialPattern>(
              downstream->stmt_pattern()) ||
          std::holds_alternative<AnchorPattern>(downstream->stmt_pattern());

      if (can_fuse) {
        VLOG(4) << "\ndownstream [" << i << "]: " << downstream->DebugStr();
        auto merged_node = graph->MergeNode(upstream, downstream, MergePattern);
        graph->RemoveNode(downstream);
        VLOG(4) << "\nmerged [" << i << "] " << merged_node->DebugStr();
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

struct ReducePlusReduceFusionOperation {
  PatternNodePtr operator()(PatternGraph* graph, PatternNodePtr node) {
    PADDLE_ENFORCE_EQ(
        node->downstream().size(),
        1,
        ::common::errors::PreconditionNotMet(
            "The downstream of the ReduceTree node should be 1, but got %d.",
            node->downstream().size()));
    auto downstream = node->downstream().at(0);
    auto loop_sink_transform_opt = GetValidAdjacentLoopTransform(
        node->loop_axis_mapping(), downstream->loop_axis_mapping(), false);
    if (loop_sink_transform_opt == std::nullopt) return node;
    auto loop_sink_transform = InsertSubstituteReduceAxis(
        loop_sink_transform_opt.value(), node->loop_axis_mapping());
    VLOG(4) << "Start AnchorFusionOperation";
    VLOG(4) << "Upstream: \n" << node->DebugStr();
    VLOG(4) << "Downstream: \n" << downstream->DebugStr();
    const auto merge_pattern_fn =
        [](const StmtPattern& upstream,
           const StmtPattern& downstream) -> StmtPattern {
      return AnchorPattern(
          UniqueConcatVector(GetOpsInPattern(upstream),
                             GetOpsInPattern(downstream)),
          std::make_shared<FusionTracker>(GetFusionTracker(upstream),
                                          GetFusionTracker(downstream)),
          LoopAxisMappingMerge(GetPatternLoopAxisMapping(upstream),
                               GetPatternLoopAxisMapping(downstream),
                               false));
    };
    auto merged_node = graph->MergeNode(node, downstream, merge_pattern_fn);
    // Update tracker
    auto node_tmp_id = GetNewTmpId(node->id());
    merged_node->AppendInstr(std::make_shared<AxisTransformInstr>(
        node->id(), node_tmp_id, loop_sink_transform));
    merged_node->AppendInstr(std::make_shared<CombineInstr>(
        std::vector<std::string>{node_tmp_id, downstream->id()},
        merged_node->id()));
    graph->RemoveNode(downstream);
    graph->RemoveNode(node);
    VLOG(4) << "Merged: \n" << merged_node->DebugStr();
    merged_node->UpdateTracker();
    return merged_node;
  }
};

struct ReducePlusTrivialFusionOperation {
  PatternNodePtr operator()(PatternGraph* graph, PatternNodePtr node) {
    PADDLE_ENFORCE_EQ(
        node->downstream().size(),
        1,
        ::common::errors::PreconditionNotMet(
            "The downstream of the ReduceTree node should be 1, but got %d.",
            node->downstream().size()));
    auto downstream = node->downstream().at(0);
    auto loop_transform_pair = GetReducePlusTrivialLoopTransform(
        node->loop_axis_mapping(), downstream->loop_axis_mapping());
    if (loop_transform_pair == std::nullopt) return node;
    VLOG(4) << "Start AnchorFusionOperation";
    VLOG(4) << "Upstream: \n" << node->DebugStr();
    VLOG(4) << "Downstream: \n" << downstream->DebugStr();
    auto upstream_loop_transform = InsertSubstituteReduceAxis(
        loop_transform_pair.value().first, node->loop_axis_mapping());
    auto downstream_loop_transform = loop_transform_pair.value().second;
    const auto merge_pattern_fn =
        [&](const StmtPattern& upstream,
            const StmtPattern& downstream) -> StmtPattern {
      return AnchorPattern(
          UniqueConcatVector(GetOpsInPattern(upstream),
                             GetOpsInPattern(downstream)),
          std::make_shared<FusionTracker>(GetFusionTracker(upstream),
                                          GetFusionTracker(downstream)),
          ReducePlusTrivialLoopAxisMappingMerge(
              GetPatternLoopAxisMapping(upstream),
              GetPatternLoopAxisMapping(downstream),
              downstream_loop_transform));
    };
    auto merged_node = graph->MergeNode(node, downstream, merge_pattern_fn);
    // Update tracker
    auto node_tmp_id = GetNewTmpId(node->id());
    auto downstream_tmp_id = GetNewTmpId(downstream->id());
    merged_node->AppendInstr(std::make_shared<AxisTransformInstr>(
        node->id(), node_tmp_id, upstream_loop_transform));
    merged_node->AppendInstr(std::make_shared<AxisTransformInstr>(
        downstream->id(), downstream_tmp_id, downstream_loop_transform));
    merged_node->AppendInstr(std::make_shared<CombineInstr>(
        std::vector<std::string>{node_tmp_id, downstream_tmp_id},
        merged_node->id()));
    graph->RemoveNode(downstream);
    graph->RemoveNode(node);
    VLOG(4) << "Merged: \n" << merged_node->DebugStr();
    merged_node->UpdateTracker();
    return merged_node;
  }
};

struct LiftReduceToReduceTreeOperation {
  PatternNodePtr operator()(PatternGraph* graph, PatternNodePtr node) {
    auto origin_name = node->id();
    const auto& reduce_pattern = std::get<ReducePattern>(node->stmt_pattern());
    const auto& loop_axis_mapping = node->loop_axis_mapping();
    node->set_stmt_pattern(ReduceTreePattern(
        {},
        reduce_pattern,
        std::make_shared<FusionTracker>(reduce_pattern.tracker_)));
    node->set_loop_axis_mapping(loop_axis_mapping);
    node->AppendInstr(std::make_shared<CopyInstr>(origin_name, node->id()));
    VLOG(4) << "Make CopyInstr: " << origin_name << " -> " << node->id();
    return node;
  }
};

struct LiftToAnchorPatternOperation {
  PatternNodePtr operator()(PatternGraph* graph, PatternNodePtr node) {
    std::string origin_name = node->id();
    node->set_stmt_pattern(AnchorPattern(
        GetOpsInPattern(node->stmt_pattern()),
        std::make_shared<FusionTracker>(GetFusionTracker(node->stmt_pattern())),
        node->loop_axis_mapping()));
    node->AppendInstr(std::make_shared<CopyInstr>(origin_name, node->id()));
    VLOG(4) << "Make CopyInstr: " << origin_name << " -> " << node->id();
    return node;
  }
};

struct AnchorFusionOperation {
  PatternNodePtr operator()(PatternGraph* graph,
                            const PatternNodePtr& upstream,
                            const PatternNodePtr& downstream) {
    bool upstream_is_anchor;
    AxisTransformRoute loop_transform;
    auto loop_lift_transform = GetValidAdjacentLoopTransform(
        upstream->loop_axis_mapping(), downstream->loop_axis_mapping(), true);
    if (loop_lift_transform.has_value()) {
      loop_transform = loop_lift_transform.value();
      upstream_is_anchor = true;
    } else {
      auto loop_sink_transform =
          GetValidAdjacentLoopTransform(upstream->loop_axis_mapping(),
                                        downstream->loop_axis_mapping(),
                                        false);
      if (!loop_sink_transform.has_value()) {
        return upstream;
      }
      loop_transform = loop_sink_transform.value();
      upstream_is_anchor = false;
    }
    VLOG(4) << "Start AnchorFusionOperation";
    VLOG(4) << "Upstream: \n" << upstream->DebugStr();
    VLOG(4) << "Downstream: \n" << downstream->DebugStr();
    const auto merge_pattern_fn =
        [upstream_is_anchor](const StmtPattern& upstream,
                             const StmtPattern& downstream) -> StmtPattern {
      return AnchorPattern(
          UniqueConcatVector(GetOpsInPattern(upstream),
                             GetOpsInPattern(downstream)),
          std::make_shared<FusionTracker>(GetFusionTracker(upstream),
                                          GetFusionTracker(downstream)),
          LoopAxisMappingMerge(GetPatternLoopAxisMapping(upstream),
                               GetPatternLoopAxisMapping(downstream),
                               upstream_is_anchor));
    };
    auto merged_node = graph->MergeNode(upstream, downstream, merge_pattern_fn);
    // Update tracker
    if (upstream_is_anchor) {
      auto downstream_tmp_id = GetNewTmpId(downstream->id());
      merged_node->AppendInstr(std::make_shared<AxisTransformInstr>(
          downstream->id(), downstream_tmp_id, loop_transform));
      merged_node->AppendInstr(std::make_shared<CombineInstr>(
          std::vector<std::string>{upstream->id(), downstream_tmp_id},
          merged_node->id()));
    } else {
      auto upstream_tmp_id = GetNewTmpId(upstream->id());
      merged_node->AppendInstr(std::make_shared<AxisTransformInstr>(
          upstream->id(), upstream_tmp_id, loop_transform));
      merged_node->AppendInstr(std::make_shared<CombineInstr>(
          std::vector<std::string>{upstream_tmp_id, downstream->id()},
          merged_node->id()));
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
            std::get<AnchorPattern>(upstream->stmt_pattern()).tracker_));
    auto loop_axis_mapping = upstream->loop_axis_mapping();
    upstream->set_stmt_pattern(trivial_pattern);
    upstream->set_loop_axis_mapping(loop_axis_mapping);
    VLOG(4) << "Make CopyInstr: " << origin_name << " -> " << upstream->id();
    upstream->AppendInstr(
        std::make_shared<CopyInstr>(origin_name, upstream->id()));
    MergeTrivialPatternOperation()(graph, upstream);
  }
};

struct HorizontalFusionOperation {
  PatternNodePtr operator()(PatternGraph* graph,
                            const PatternNodePtr& lhs,
                            const PatternNodePtr& rhs) {
    AxisTransformRoute loop_transform;
    bool lhs_is_anchor;
    auto rhs_to_lhs_transform = GetValidHorizontalLoopTransform(
        rhs->loop_axis_mapping(), lhs->loop_axis_mapping());
    if (rhs_to_lhs_transform.has_value()) {
      lhs_is_anchor = true;
      loop_transform = rhs_to_lhs_transform.value();
    } else {
      auto lhs_to_rhs_transform = GetValidHorizontalLoopTransform(
          lhs->loop_axis_mapping(), rhs->loop_axis_mapping());
      if (!lhs_to_rhs_transform.has_value()) return nullptr;
      lhs_is_anchor = false;
      loop_transform = lhs_to_rhs_transform.value();
    }
    auto source = lhs_is_anchor ? rhs : lhs;
    auto target = lhs_is_anchor ? lhs : rhs;
    VLOG(4) << "Start HorizontalFusionOperation from " << source->id() << " to "
            << target->id();
    VLOG(4) << "source: \n" << source->DebugStr();
    VLOG(4) << "target: \n" << target->DebugStr();
    const auto merge_pattern_fn = [](const StmtPattern& source,
                                     const StmtPattern& target) -> StmtPattern {
      return AnchorPattern(
          UniqueConcatVector(GetOpsInPattern(source), GetOpsInPattern(target)),
          std::make_shared<FusionTracker>(GetFusionTracker(source),
                                          GetFusionTracker(target)),
          HorizontalLoopAxisMappingMerge(GetPatternLoopAxisMapping(source),
                                         GetPatternLoopAxisMapping(target)));
    };
    auto merged_node = graph->MergeNode(source, target, merge_pattern_fn);
    auto source_tmp_id = GetNewTmpId(source->id());
    merged_node->AppendInstr(std::make_shared<AxisTransformInstr>(
        source->id(), source_tmp_id, loop_transform));
    merged_node->AppendInstr(std::make_shared<CombineInstr>(
        std::vector<std::string>{source_tmp_id, target->id()},
        merged_node->id()));
    graph->RemoveNode(source);
    graph->RemoveNode(target);
    VLOG(4) << "Merged: \n" << merged_node->DebugStr();
    return merged_node;
  }
};

}  // namespace cinn::fusion
