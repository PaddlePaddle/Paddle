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
           node->downstream().at(0)->downstream().size() == 0 &&
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
        !StmtPatternGraphMatcher<ReduceTreePlusTrivialPattern>()(graph, node) &&
        !StmtPatternGraphMatcher<ReducePattern>()(graph, node);
    // TODO(huangjiyi): Support anchor value is reduce output.
    // bool reduce_tree_with_single_reduce =
    //     StmtPatternGraphMatcher<ReduceTreePattern>()(graph, node) &&
    //     std::get<ReduceTreePattern>(node->stmt_pattern()).childs().size() ==
    //     0;
    return not_reduce_tree /* || reduce_tree_with_single_reduce */;
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

struct LEOneElementWiseDownstreamMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    size_t count = 0;
    for (const auto& downsteram : node->downstream()) {
      if (StmtPatternGraphMatcher<TrivialPattern>()(graph, downsteram)) {
        auto ops = std::get<TrivialPattern>(downsteram->stmt_pattern()).ops();
        bool is_elementwise =
            std::all_of(ops.begin(), ops.end(), [](pir::Operation* op) {
              return GetOpPatternKind(op) == hlir::framework::kElementWise;
            });
        count += is_elementwise;
      }
    }
    return (count <= 1);
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
  bool operator()(const PatternGraph& graph,
                  const PatternNodePtr& lhs,
                  const PatternNodePtr& rhs) {
    return A()(graph, lhs, rhs);
  }
};

template <typename A, typename... Args>
struct And<A, Args...> {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return A()(graph, node) && And<Args...>()(graph, node);
  }
  bool operator()(const PatternGraph& graph,
                  const PatternNodePtr& lhs,
                  const PatternNodePtr& rhs) {
    return A()(graph, lhs, rhs) && And<Args...>()(graph, lhs, rhs);
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

struct HorizontalFusionConstrain {
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

    return graph.policy_manager().GetPolicy<GeneralTopoPolicy>()->CanFuse(
               lhs, rhs) &&
           IsLoopFrameworkEqual(lhs_pattern.padding_patterns_.back().pattern,
                                rhs_pattern.padding_patterns_.back().pattern);
  }
};

/*
 * We must limit the output + input + shape_info number and make sure
 * the number is smaller than 512.
 */
struct InputOutputMaximumConstrain {
  const int MAX_INPUT_OUTPUT_NUMBER = 480;  // cuda only support 512
  std::vector<pir::Value> GetInputValuesExceptMiddle(
      const std::vector<pir::Operation*>& ops) {
    return VectorDiff(GetInputsValue(ops), GetOutputsValue(ops));
  }
  std::vector<pir::Value> GetOutputValuesExceptMiddle(
      const std::vector<pir::Operation*>& ops) {
    return VectorDiff(GetOutputsValue(ops), GetInputsValue(ops));
  }
  std::vector<pir::Operation*> GetAllOps(const PatternNodePtr& lhs,
                                         const PatternNodePtr& rhs) {
    return UniqueVectorBySet(
        ConcatVector(GetOpsInPattern(lhs->stmt_pattern()),
                     GetOpsInPattern(rhs->stmt_pattern())));
  }
  bool operator()(const PatternGraph& graph,
                  const PatternNodePtr& lhs,
                  const PatternNodePtr& rhs) {
    const auto& all_ops = GetAllOps(lhs, rhs);
    int input_number = GetInputValuesExceptMiddle(all_ops).size();
    int output_number = GetOutputValuesExceptMiddle(all_ops).size();
    return input_number + output_number < MAX_INPUT_OUTPUT_NUMBER;
  }
};

struct HorizontalCheckMiddleOutputVar {
  bool DontHaveMiddleVariable(const PatternGraph& graph,
                              const PatternNodePtr& lhs,
                              const PatternNodePtr& rhs) {
    for (const auto& i : lhs->downstream()) {
      if (i == rhs) return false;
    }
    for (const auto& i : lhs->upstream()) {
      if (i == rhs) return false;
    }
    return true;
  }

  std::vector<ValueDim> SqueezedValueDim(const LoopValueDims& vdims) {
    return FilterVector(vdims, [](const ValueDim& v) {
      return !v.empty() && GetDimExprsFromValue(v.v_)[v.idx_] != 1;
    });
  }

  bool IdenticalDep(const PatternGraph& graph,
                    const LoopValueDims& lhs_dims,
                    const LoopValueDims& rhs_dims) {
    auto sp = graph.policy_manager().template GetPolicy<RelativeJudgePolicy>();
    auto get_axes_from_valuedim = [&](const ValueDim& vdim) {
      return (sp->GetAxesInfoManager()).GetAxes(vdim.v_).axis_names[vdim.idx_];
    };
    VLOG(4) << "origin lhs_dims.size() = " << lhs_dims.size();
    VLOG(4) << "origin rhs_dims.size() = " << rhs_dims.size();
    std::vector<ValueDim> lhs_squeeze_value_dim = SqueezedValueDim(lhs_dims);
    std::vector<ValueDim> rhs_squeeze_value_dim = SqueezedValueDim(rhs_dims);

    if (VLOG_IS_ON(4)) {
      VLOG(4) << "lhs_squeeze_value_dim is : ";
      for (int i = 0; i < lhs_squeeze_value_dim.size(); ++i) {
        VLOG(4) << "    " << i << " = " << lhs_squeeze_value_dim[i].DebugStr();
        VLOG(4) << "    "
                << "shardable axes: "
                << get_axes_from_valuedim(lhs_squeeze_value_dim[i]);
      }
      VLOG(4) << "lhs_squeeze_value_dim is : ";
      if (VLOG_IS_ON(4)) {
        for (int i = 0; i < rhs_squeeze_value_dim.size(); ++i) {
          VLOG(4) << "    " << i << " = "
                  << rhs_squeeze_value_dim[i].DebugStr();
          VLOG(4) << "    "
                  << "shardable axes: "
                  << get_axes_from_valuedim(rhs_squeeze_value_dim[i]);
        }
      }
    }

    // compare non_one value dims of
    PADDLE_ENFORCE_EQ(lhs_squeeze_value_dim.size(),
                      rhs_squeeze_value_dim.size(),
                      "lhs squeezed dims is not equal to rhs squeezed dims");
    for (int i = 0; i < lhs_squeeze_value_dim.size(); ++i) {
      if (get_axes_from_valuedim(lhs_squeeze_value_dim[i]) !=
          get_axes_from_valuedim(rhs_squeeze_value_dim[i]))
        return false;
    }
    return true;
  }
  bool IdenticalDepAll(const PatternGraph& graph,
                       const LoopValueDims& rhs_dims,
                       const std::vector<LoopValueDims> lhs_dims_vec) {
    std::function<bool(const LoopValueDims)> is_identical_dep =
        [&](const LoopValueDims out) {
          return IdenticalDep(graph, rhs_dims, out);
        };
    return All(MapVector(lhs_dims_vec, is_identical_dep));
  }
  bool operator()(const PatternGraph& graph,
                  const PatternNodePtr& lhs,
                  const PatternNodePtr& rhs) {
    // Middle Variable Must be ( id-dependent ) to support horizontal fusion.
    if (DontHaveMiddleVariable(graph, lhs, rhs)) return true;
    const auto& left_dims_vec = GetLoopValueDims(lhs->stmt_pattern());
    const auto& right_dims_vec = GetLoopValueDims(rhs->stmt_pattern());
    bool identical_dep = true;
    for (const auto& right_dims : right_dims_vec) {
      identical_dep &= IdenticalDepAll(graph, right_dims, left_dims_vec);
    }
    return identical_dep;
  }
};

}  // namespace cinn::fusion
