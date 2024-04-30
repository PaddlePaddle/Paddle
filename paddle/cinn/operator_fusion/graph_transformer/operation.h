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

// Operation

struct MergeReduceTreeOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph, PatternNodePtr<Phrase> node) {
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
  }
};

struct MergeReduceTreeAndTrivialOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph, PatternNodePtr<Phrase> node) {
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
  }
};

struct LiftReduceToReduceTreeOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph, PatternNodePtr<Phrase> node) {
    const auto& reduce_pattern = ToReducePattern<Phrase>(node->stmt_pattern());
    node->set_stmt_pattern(ReduceTreePattern<Phrase>({}, reduce_pattern));
    VLOG(4) << "LiftReduceToReduceTreeOperation: \nnode " << node->DebugStr();
  }
};

struct MergeTrivialPatternOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph, PatternNodePtr<Phrase> node) {
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
  }
};

struct LiftToHorizontalFusionPatternOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph, PatternNodePtr<Phrase> node) {
    node->set_stmt_pattern(
        HorizontalFusionPattern<Phrase>({node->stmt_pattern()}));
  }
};

struct LiftToAnchorPatternOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph, PatternNodePtr<Phrase> node) {
    std::vector<pir::Operation*> ops = GetOpsInPattern(pattern);
    pir::Value anchor = node->sink_op()->result(0);
    node->set_stmt_pattern(AnchorPattern<Phrase>(ops, anchor));
  }
};

struct FuseUpstreamAnchorOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph,
                  const PatternNodePtr<Phrase>& upstream,
                  const PatternNodePtr<Phrase>& downstream) {
    // TODO(@wuzhanfei)
  }
};

struct FuseDownstreamAnchorOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph,
                  const PatternNodePtr<Phrase>& upstream,
                  const PatternNodePtr<Phrase>& downstream) {
    // TODO(@wuzhanfei)
  }
};

struct SplitRecomputeOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph,
                  PatternNodePtr<Phrase> upstream) {
    PADDLE_ENFORCE_GT(upstream->downstream().size(),
                      1,
                      phi::errors::PreconditionNotMet(
                          "The downstream of node for recomputation should be "
                          "more than 1, but got %d.",
                          upstream->downstream().size()));

    std::vector<PatternNodePtr<Phrase>> fusion_candidate =
        upstream->downstream();
    upstream->ClearDownstream();

    for (const auto& downstream : fusion_candidate) {
      bool can_fuse = graph->policy_manager()
                          .template GetPolicy<AnchorSearchPolicy>()
                          ->HasDownstreamAnchor(upstream, downstream) ||
                      graph->policy_manager()
                          .template GetPolicy<AnchorSearchPolicy>()
                          ->HasUpstreamAnchor(upstream, downstream);
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
  void operator()(PatternGraph<Phrase>* graph,
                  const PatternNodePtr<Phrase>& i,
                  const PatternNodePtr<Phrase>& j) {
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
    graph->MergeNode(i, j, MergePattern<Phrase>);
    graph->RemoveNode(i);
    graph->RemoveNode(j);
  }
};

}  // namespace cinn::fusion
