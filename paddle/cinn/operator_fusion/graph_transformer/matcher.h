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

struct NonSinkNodeMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return !node->downstream().empty();
  }
};

struct IsOutputNodeMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    for (auto op : node->ops()) {
      if (graph.output_ops().count(op)) return true;
    }
    return false;
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

struct OnlyOneDownstreamMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return node->downstream().size() == 1;
  }
};

/*
 * We must limit the output + input + shape_info number and make sure
 * the number is smaller than 512.
 */
struct InputOutputMaximumConstrain {
  const int MAX_INPUT_OUTPUT_NUMBER = 384;  // cuda only support 512
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

template <typename StmtPattern>
struct StmtPatternGraphMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return GetPatternType(node->stmt_pattern()) == StmtPattern::type();
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

struct CanAnchorFusionMatcher {
  bool operator()(const PatternGraph& graph,
                  const PatternNodePtr& upstream,
                  const PatternNodePtr& downstream) {
    return StmtPatternGraphMatcher<AnchorPattern>()(graph, upstream) &&
           StmtPatternGraphMatcher<AnchorPattern>()(graph, downstream) &&
           graph.policy_manager()
               .template GetPolicy<GeneralTopoPolicy>()
               ->CanFuse(upstream, downstream);
  }
};

struct RecomputeNodeMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    const auto can_recompute_fn = [](const PatternNodePtr& node) -> bool {
      // Current Algorithm:
      // An node can be recomputed if:
      // 1. It didn't go through any pattern merging during prior fusions,
      // which means it only has one output value.
      // 2. It only contains trivial ops.
      if (node->loop_axis_mapping().output_values.size() > 1) {
        return false;
      }
      bool has_combine_fusion =
          std::any_of(node->fusion_tracker()->instructions_.begin(),
                      node->fusion_tracker()->instructions_.end(),
                      [](const FusionInstrPtr& instr) {
                        return instr->type() == T_Combine;
                      });
      if (has_combine_fusion) {
        return false;
      }
      for (const auto& op : GetOpsInPattern(node->stmt_pattern())) {
        const auto& op_kind = GetOpPatternKind(op);
        if (op_kind >= hlir::framework::kReduction) {
          return false;
        }
      }
      return true;
    };

    const auto input_output_nums_constraint = [](const PatternGraph& graph,
                                                 const PatternNodePtr& node) {
      return std::all_of(node->downstream().begin(),
                         node->downstream().end(),
                         [&](const PatternNodePtr& downstream) {
                           return InputOutputMaximumConstrain()(
                               graph, node, downstream);
                         });
    };

    return StmtPatternGraphMatcher<AnchorPattern>()(graph, node) &&
           !IsOutputNodeMatcher()(graph, node) &&
           node->downstream().size() >= 1 && can_recompute_fn(node) &&
           input_output_nums_constraint(graph, node);
  }
};

struct TransposeOpMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return (node->sink_op()->name() == "pd_op.transpose");
  }
};

struct NotAllElementWiseDownstreamMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    size_t count = 0;
    for (const auto& downstream : node->downstream()) {
      if (StmtPatternGraphMatcher<TrivialPattern>()(graph, downstream)) {
        auto ops = std::get<TrivialPattern>(downstream->stmt_pattern()).ops();
        bool is_elementwise =
            std::all_of(ops.begin(), ops.end(), [](pir::Operation* op) {
              return GetOpPatternKind(op) == hlir::framework::kElementWise;
            });
        count += is_elementwise;
      }
    }
    return (count < node->downstream().size());
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
  bool IsAdjacentRelation(const LoopAxisMapping& lhs,
                          const LoopAxisMapping& rhs) {
    return AnyFirstInSecond(lhs.output_values, rhs.input_values) ||
           AnyFirstInSecond(rhs.output_values, lhs.input_values);
  }

  bool MemoryIncreaseConstraint(const LoopAxisMapping& lhs,
                                const LoopAxisMapping& rhs) {
    static const std::int64_t MEMORY_INCREASE_LIMIT =
        8 * 1024 * 1024 * 64;  // 64 MB

    const auto memory_size_of_input_values =
        [](const std::vector<pir::Value>& values) -> std::int64_t {
      std::int64_t memory_size = 0;
      for (const auto& v : values) {
        const auto shape_product =
            GetShapeProduct(GetCompatibleValueAllDims(v));
        if (shape_product.isa<std::int64_t>()) {
          memory_size += shape_product.dyn_cast<std::int64_t>() * 32;
        } else {
          // Dynamic shape is not supported yet.
          return -1;
        }
      }
      return memory_size;
    };

    const auto& [_unused1, lhs_unique_input_values] =
        SplitFirstWhetherInSecond(lhs.input_values, rhs.input_values);
    const auto& [_unused2, rhs_unique_input_values] =
        SplitFirstWhetherInSecond(rhs.input_values, lhs.input_values);
    std::int64_t lhs_memory_size =
        memory_size_of_input_values(lhs_unique_input_values);
    std::int64_t rhs_memory_size =
        memory_size_of_input_values(rhs_unique_input_values);
    std::int64_t memory_increase_size =
        (lhs_memory_size + rhs_memory_size) -
        std::max(lhs_memory_size, rhs_memory_size);

    if (memory_increase_size < MEMORY_INCREASE_LIMIT) {
      return true;
    } else {
      VLOG(4) << "Can not horizontal fusion due to memory may increase "
              << memory_increase_size / 1024 / 1024 / 8 << " MB"
              << ", which exceeds the limit = 64 MB";
      return false;
    }
  }

  bool IsLoopFrameworkEqual(const LoopAxisMapping& lhs,
                            const LoopAxisMapping& rhs) {
    if (lhs.loop.empty() || rhs.loop.empty()) return false;

    const auto lhs_reduce_loop = SliceVector(
        lhs.loop, lhs.loop.size() - lhs.reduce_axis_num, lhs.loop.size());
    const auto rhs_reduce_loop = SliceVector(
        rhs.loop, rhs.loop.size() - rhs.reduce_axis_num, rhs.loop.size());

    bool reduce_equal = lhs_reduce_loop.empty() || rhs_reduce_loop.empty() ||
                        lhs_reduce_loop == rhs_reduce_loop;

    return reduce_equal && ShapeProductEqual(lhs.loop, rhs.loop);
  }

  bool operator()(const PatternGraph& graph,
                  const PatternNodePtr& lhs,
                  const PatternNodePtr& rhs) {
    return StmtPatternGraphMatcher<AnchorPattern>()(graph, lhs) &&
           StmtPatternGraphMatcher<AnchorPattern>()(graph, rhs) &&
           graph.policy_manager().GetPolicy<GeneralTopoPolicy>()->CanFuse(
               lhs, rhs) &&
           MemoryIncreaseConstraint(lhs->loop_axis_mapping(),
                                    rhs->loop_axis_mapping()) &&
           !IsAdjacentRelation(lhs->loop_axis_mapping(),
                               rhs->loop_axis_mapping()) &&
           IsLoopFrameworkEqual(lhs->loop_axis_mapping(),
                                rhs->loop_axis_mapping());
  }
};

}  // namespace cinn::fusion
