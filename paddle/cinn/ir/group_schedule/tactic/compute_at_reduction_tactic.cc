// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/group_schedule/tactic/compute_at_reduction_tactic.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"

namespace cinn {
namespace ir {
namespace {

/**
 * Do loop fusion using the SimpleComputeAt primitive.
 *
 * This tactic fuses loops that can be executed parallelly and have common
 * loads, in order to eliminate duplicate memory reads. For each schedule block
 * in the graph, it uses a four-step procedure (see FindCandidateBlocks for
 * details) to find candidate blocks that are both VALID and BENEFICIAL to do
 * ComputeAt. If found, it picks the best candidate and fuses the current block
 * to the candidate block.
 *
 * This tactic currently supports 3 fusion patterns:
 *    Reduce -> Reduce
 *    Trivial -> Reduce
 *    Trivial -> Trivial
 * We don't support `Reduce -> Trivial` because we treat Reduce as the anchor,
 * so we should move Trivial instead of moving Reduce.
 */
class ComputeAtReductionTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context, ir::IRSchedule* sch) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override { return "ComputeAtReductionTactic"; }

 private:
  /**
   * Find the candidate blocks that are both valid and beneficial for this block
   * to do ComputeAt with.
   *
   * Note:
   * 1) We only do ComputeAt in the direction `this_block -> candidate_block`.
   *    If this direction is not applicable but the opposite direction does,
   *    ComputeAt will be performed when the outer Scheduler applies on that
   *    candidate block.
   * 2) This interface currently only returns at most one block, so if there are
   *    multiple candidates, it choose one of them using some heuristics.
   */
  std::vector<std::string> FindCandidateBlocks(ir::IRSchedule* sch,
                                               const std::string& block_id);

  /**
   * Get blocks in this graph to which we can fuse the current block without
   * dependency harzards.
   *
   * A block will not cause dependency harzard with the current block if:
   * 1) It has no data dependency with the current block, and
   * 2) There is no block that would cause dependency harzard when we move the
   *    current block to the target block.
   */
  std::vector<std::string> GetDependencyHarzardFreeBlocks(
      ir::IRSchedule* sch, const std::string& block_id);

  /**
   * Get the control flow (including For and IfThenElse) of this block.
   * The loop_vars in the returned nodes are unifiedly rewritten to the form
   * `$<loop_index>` to facilitate comparison.
   */
  std::vector<ir::Expr> GetUnifiedControlFlow(
      const ir::Expr& block, const std::vector<ir::Expr>& loops);

  /**
   * Get loads in this schedule block whose indices correlate with the serial
   * loop's index. Only with these loads can we save memory accesses by doing
   * loop fusion.
   * The loop_vars in the returned loads' indices are unifiedly rewritten to the
   * form `$<loop_index>` to facilitate comparison.
   */
  std::vector<ir::Expr> GetLoopVariantLoads(const ir::Expr& block,
                                            const std::vector<ir::Expr>& loops);

  /**
   * Get the total extent of serial loops (loops not bound on GPU block/thread).
   * Returns -1 if the extent is dynamic.
   */
  int64_t GetSerialLoopExtent(const std::vector<ir::Expr>& loops);

 private:
  ScheduleContext* context_;

  // A copy of the IRSchedule and ScheduleBlockGraph, with all loop vars
  // unifiedly rewritten to the form `$<loop_index>` (e.g. $0, $1).
  std::unique_ptr<ir::IRSchedule> sch_;
  std::unique_ptr<ir::ScheduleBlockGraph> graph_;

  // Cache of the results of GetUnifiedControlFlow, GetLoopVariantLoads,
  // GetSerialLoopExtent and IsReductionSBlock because these functions are too
  // slow while their results don't change after scheduling.
  std::unordered_map<std::string, std::vector<ir::Expr>> unified_control_flow_;
  std::unordered_map<std::string, std::vector<ir::Expr>> loop_variant_loads_;
  std::unordered_map<std::string, int64_t> serial_loop_extent_;
  std::unordered_map<std::string, bool> is_reduction_sblock_;
};

bool ControlFlowAllEqual(const std::vector<ir::Expr>& first,
                         const std::vector<ir::Expr>& second) {
  // Check if without false case and condition is an index expression, only
  // ir::LT, ir::LE, now
  const auto IsIndexCondWithoutFalseCase =
      [&](const ir::IfThenElse* if_op) -> bool {
    if (if_op->false_case.defined()) return false;
    auto cond = if_op->condition;
    if (cond.As<ir::LT>()) {
      auto lt = cond.As<ir::LT>();
      return lt->a().is_index() && lt->b().is_index();
    }
    if (cond.As<ir::LE>()) {
      auto le = cond.As<ir::LE>();
      return le->a().is_index() && le->b().is_index();
    }
    return false;
  };

  const auto ControlFlowEqual = [&](const ir::Expr& first,
                                    const ir::Expr& second) -> bool {
    if (first.As<ir::For>() && second.As<ir::For>()) {
      auto first_for = first.As<ir::For>();
      auto second_for = second.As<ir::For>();
      if (first_for->for_type() != second_for->for_type()) return false;
      return first_for->extent == second_for->extent;
    } else if (first.As<ir::IfThenElse>() && second.As<ir::IfThenElse>()) {
      auto first_if = first.As<ir::IfThenElse>();
      auto second_if = second.As<ir::IfThenElse>();
      if (!IsIndexCondWithoutFalseCase(first_if)) return false;
      if (!IsIndexCondWithoutFalseCase(second_if)) return false;
      return first_if->condition == second_if->condition;
    } else {
      return false;
    }
    return false;
  };

  if (first.size() != second.size()) return false;
  for (size_t i = 0; i < first.size(); ++i) {
    if (!ControlFlowEqual(first[i], second[i])) return false;
  }
  return true;
}

bool BlockWithSameLoop(const std::vector<ir::Expr>& first,
                       const std::vector<ir::Expr>& second) {
  VLOG(8) << "First inner loop: " << first.back();
  VLOG(8) << "Second inner loop: " << second.back();
  VLOG(8) << "Equal: " << (first.back() == second.back());
  return first.back() == second.back();
}

bool HasCommonLoad(const std::vector<ir::Expr>& first_loads,
                   const std::vector<ir::Expr>& other_loads) {
  for (auto& first_load : first_loads) {
    for (auto& other_load : other_loads) {
      if (ir::ir_utils::IRCompare(first_load, other_load)) return true;
    }
  }
  return false;
}

bool LoadsAllEqual(const std::vector<ir::Expr>& first_loads,
                   const std::vector<ir::Expr>& other_loads) {
  if (first_loads.size() != other_loads.size()) return false;
  for (auto& load : first_loads) {
    if (!HasCommonLoad({load}, other_loads)) return false;
  }
  return true;
}

struct GetControlFlowFunctor {
  explicit GetControlFlowFunctor(const Expr& block) : block_(block) {}

  std::vector<Expr> operator()(const Expr& expr) {
    PADDLE_ENFORCE_NOT_NULL(
        block_.As<ir::ScheduleBlockRealize>(),
        ::common::errors::NotFound("The expr should be ScheduleBlockRealize."));
    end_ = false;
    GetControlFlow(expr);
    return result_;
  }

 private:
  void GetControlFlow(const Expr& expr) {
    if (end_) return;
    if (expr.As<ir::For>()) {
      control_flow_.emplace_back(expr);
      GetControlFlow(expr.As<ir::For>()->body);
      control_flow_.pop_back();
    } else if (expr.As<ir::ScheduleBlockRealize>()) {
      if (analyzer::GetBlockName(expr) == analyzer::GetBlockName(block_)) {
        result_ = control_flow_;
        end_ = true;
        return;
      } else {
        GetControlFlow(expr.As<ir::ScheduleBlockRealize>()->schedule_block);
      }
    } else if (expr.As<ir::ScheduleBlock>()) {
      GetControlFlow(expr.As<ir::ScheduleBlock>()->body);
    } else if (expr.As<ir::Block>()) {
      for (auto& stmt : expr.As<ir::Block>()->stmts) GetControlFlow(stmt);
    } else if (expr.As<ir::IfThenElse>()) {
      control_flow_.emplace_back(expr);
      GetControlFlow(expr.As<ir::IfThenElse>()->true_case);
      if (expr.As<ir::IfThenElse>()->false_case.defined())
        GetControlFlow(expr.As<ir::IfThenElse>()->false_case);
      control_flow_.pop_back();
    }
  }

  std::vector<Expr> control_flow_{};
  std::vector<Expr> result_{};
  bool end_{false};
  const Expr& block_;
};

void ComputeAtReductionTactic::Init(ScheduleContext* context,
                                    ir::IRSchedule* sch) {
  context_ = context;
  unified_control_flow_.clear();
  loop_variant_loads_.clear();
  serial_loop_extent_.clear();

  sch_ = std::make_unique<ir::IRSchedule>(*sch);
  graph_ = std::make_unique<ir::ScheduleBlockGraph>(*sch_);

  for (auto& block : sch_->GetAllBlocks()) {
    // Replace loop_vars to the unified form `$<loop_index>`
    std::vector<ir::Expr> loops = sch_->GetLoops(block);
    for (int i = 0; i < loops.size(); ++i) {
      auto& loop_var = loops[i].As<ir::For>()->loop_var;
      ir::Expr unified_var(ir::Var("$" + std::to_string(i)));
      optim::ReplaceVarWithExpr(&loops[i], loop_var, unified_var);
    }

    // Replace iter_vars to iter_values in schedule block
    auto* block_realize = block.As<ir::ScheduleBlockRealize>();
    auto* block_node = block_realize->schedule_block.As<ir::ScheduleBlock>();
    block_node->body = analyzer::ReplaceVarWithExpr(
        block_node->body, block_node->iter_vars, block_realize->iter_values);

    // Cache the results of some slow functions to accelerate Apply execution
    unified_control_flow_[block_node->name] =
        GetUnifiedControlFlow(block, loops);
    loop_variant_loads_[block_node->name] = GetLoopVariantLoads(block, loops);
    serial_loop_extent_[block_node->name] = GetSerialLoopExtent(loops);
    is_reduction_sblock_[block_node->name] = analyzer::IsReductionSBlock(block);
  }
}

void ComputeAtReductionTactic::Apply(ir::IRSchedule* sch,
                                     const std::string& block_id) {
  if (IsReduceInitTensorName(block_id)) return;
  if (serial_loop_extent_[block_id] == 1) return;  // no loop, nothing to fuse

  std::vector<std::string> candidates = FindCandidateBlocks(sch, block_id);
  if (candidates.empty()) return;

  auto& target_id = candidates.front();
  VLOG(4) << "ComputeAt Apply: " << block_id << " -> " << target_id;

  ir::Expr block = sch->GetBlock(block_id);
  sch->SimpleComputeAt(block, sch->GetLoops(target_id).back());

  // If the current block is a Reduce, don't forget to also do ComputeAt for its
  // `reduce_init` block.
  // TODO(liangshuhao): the IsReductionSBlock interface returns false for a unit
  // Reduce (Reduce whose loop extent is 1), so we have to use HasBlock to check
  // the existence of the `reduce_init` block.
  std::string reduce_init_block_id = GenReduceInitTensorNameOf(block_id);
  if (sch->HasBlock(reduce_init_block_id)) {
    std::vector<ir::Expr> loops = sch->GetLoops(reduce_init_block_id);
    PADDLE_ENFORCE_GT(
        loops.size(),
        0UL,
        ::common::errors::PreconditionNotMet(
            "The reduce_init schedule block must be inside loops, but found "
            "one that has no parent loop: %s",
            reduce_init_block_id));
    int loop_index = loops.size() - 1;
    ir::Expr reduce_init_block = sch->GetBlock(reduce_init_block_id);
    sch->SimpleComputeAt(reduce_init_block,
                         sch->GetLoops(target_id)[loop_index]);
  }
}

std::vector<std::string> ComputeAtReductionTactic::FindCandidateBlocks(
    ir::IRSchedule* sch, const std::string& block_id) {
  ir::Expr this_block = sch->GetBlock(block_id);
  std::vector<ir::Expr> this_loops = sch->GetLoops(block_id);
  std::vector<ir::Expr> this_cf = unified_control_flow_[block_id];
  std::vector<ir::Expr> this_loads = loop_variant_loads_[block_id];

  // Step 1. Get the blocks to which we can do ComputeAt without dependency
  //   hazards. This ensures the fundamental correctness of ComputeAt.
  std::vector<std::string> dep_free_blocks =
      GetDependencyHarzardFreeBlocks(sch, block_id);
  if (dep_free_blocks.empty()) return {};

  // Step 2. Get the blocks that have equal control flow with the current block.
  //    This makes the problem easier by preventing handling unaligned loops.
  //    Usually, fusing unaligned loops will not improve the performance, so it
  //    is OK to skip them.
  std::vector<std::string> cf_equal_blocks;
  for (auto& other_id : dep_free_blocks) {
    std::vector<ir::Expr> other_cf = unified_control_flow_[other_id];
    if (ControlFlowAllEqual(this_cf, other_cf)) {
      cf_equal_blocks.push_back(other_id);
    }
  }
  if (cf_equal_blocks.empty()) return {};

  // Step 3. Get the blocks that are beneficial to do ComputeAt.
  // Rules:
  // 1) The block should have common loop-variant loads with the current block,
  //    so that we can save memory accesses by loop fusion.
  // 2) Both blocks should not be inside the same loop, otherwise they have
  //    already been fused and we have nothing to do.
  // 3) If the current block is Reduce, the candidate block must also be Reduce,
  //    because we don't allow fusing Reduce into Trivial.
  // 4) If the current block is Trivial and its serial loop's extent is <= 8, we
  //    only choose blocks that share all the same loads, because fusing blocks
  //    that have different loads will introduce non-continuous reads, which can
  //    only be compensated when the loop is large enough.
  std::vector<std::string> beneficial_blocks;
  for (auto& other_id : cf_equal_blocks) {
    std::vector<ir::Expr> other_loads = loop_variant_loads_[other_id];
    if (!HasCommonLoad(this_loads, other_loads)) continue;
    std::vector<ir::Expr> other_loops = sch->GetLoops(other_id);
    if (BlockWithSameLoop(this_loops, other_loops)) continue;
    if (is_reduction_sblock_[block_id]) {
      if (!is_reduction_sblock_[other_id]) continue;
    } else {
      int64_t extent = serial_loop_extent_[block_id];
      if (extent <= 8 && extent != -1) {
        if (!LoadsAllEqual(this_loads, other_loads)) continue;
      }
    }
    beneficial_blocks.push_back(other_id);
  }
  if (beneficial_blocks.empty()) return {};

  // Step 4. Choose the best candidate using some heuristic rules.
  // Rules:
  // 1) If there exists Reduce, choose the first Reduce.
  // 2) Otherwise, just choose the first candidate.
  for (auto& other_id : beneficial_blocks) {
    if (is_reduction_sblock_[other_id]) return {other_id};
  }
  return beneficial_blocks;
}

std::vector<std::string>
ComputeAtReductionTactic::GetDependencyHarzardFreeBlocks(
    ir::IRSchedule* sch, const std::string& block_id) {
  std::vector<std::string> results;
  std::vector<ir::Expr> blocks = sch->GetAllBlocks();
  auto* graph_node = graph_->RetrieveNode(block_id);
  std::unordered_set<std::string> upstreams = graph_node->UpstreamNodes();
  std::unordered_set<std::string> downstreams = graph_node->DownstreamNodes();

  // Find the position of the current block in the graph, then search upwards
  // and downwards until a denepency harzard is met.
  //
  // For example, in the following graph (A-E are schedule blocks, `|` denotes
  // data dependency):
  //
  //     A      Search upwards
  //                  ^
  //     B            |
  //     |
  //     C   <- current block
  //
  //     D            |
  //     |            v
  //     E     Search downwards
  //
  // C has denepency harzard with B because it directly depends on B. C also has
  // dependency harzard with A, because if we move C to the position of A, we
  // will violate the dependency of B->C. C is only harzard-free with D and E.
  auto this_it =
      std::find_if(blocks.begin(), blocks.end(), [&](const ir::Expr& block) {
        return analyzer::GetBlockName(block) == block_id;
      });

  // Search upwards
  auto this_it_rev = std::make_reverse_iterator(this_it);
  for (auto it = this_it_rev; it != blocks.rend(); ++it) {
    std::string other_id = analyzer::GetBlockName(*it);
    // As a special case, we can ignore the `reduce_init` in front of Reduce.
    if (IsReduceInitTensorName(other_id)) continue;
    if (upstreams.count(other_id) > 0) break;
    results.push_back(other_id);
  }

  // Search downwards
  for (auto it = this_it + 1; it != blocks.end(); ++it) {
    std::string other_id = analyzer::GetBlockName(*it);
    if (downstreams.count(other_id) > 0) break;
    results.push_back(other_id);
  }

  return results;
}

std::vector<ir::Expr> ComputeAtReductionTactic::GetUnifiedControlFlow(
    const ir::Expr& block, const std::vector<ir::Expr>& loops) {
  if (loops.empty()) {
    return {};
  }
  GetControlFlowFunctor functor(block);
  return functor(loops[0]);
}

std::vector<ir::Expr> ComputeAtReductionTactic::GetLoopVariantLoads(
    const ir::Expr& block, const std::vector<ir::Expr>& loops) {
  std::unordered_set<ir::Var> serial_loop_vars;
  for (auto& loop : loops) {
    auto* node = loop.As<ir::For>();
    if (!node->is_binded()) {
      serial_loop_vars.insert(node->loop_var);
    }
  }

  auto ContainsSerialLoopVar = [&](const ir::Expr& expr) {
    std::set<ir::Expr> vars_in_expr =
        ir::ir_utils::CollectIRNodes(expr, [](const ir::Expr* x) {
          return x->as_var() && !x->as_var()->is_symbolic_constant;
        });
    for (auto& var : vars_in_expr) {
      if (serial_loop_vars.count(var.as_var_ref()) > 0) return true;
    }
    return false;
  };

  auto IsLoopVariantLoad = [&](const ir::Expr* x) {
    auto* node = x->As<ir::Load>();
    if (!node) return false;
    auto& buffer = node->tensor.as_tensor()->buffer;
    if (buffer->memory_type != ir::MemoryType::Heap) return false;
    for (auto& index : node->indices) {
      if (ContainsSerialLoopVar(index)) return true;
    }
    return false;
  };

  ir::Expr store = analyzer::GetStoreOfSBlock(block);
  std::set<ir::Expr> loads = ir::ir_utils::CollectIRNodes(
      store.As<ir::Store>()->value, IsLoopVariantLoad);

  // remove duplicate loads
  std::vector<ir::Expr> dedup_loads;
  for (auto& load : loads) {
    auto it = std::find_if(
        dedup_loads.begin(), dedup_loads.end(), [&](const ir::Expr& other) {
          return ir::ir_utils::IRCompare(load, other);
        });
    if (it == dedup_loads.end()) {
      dedup_loads.push_back(load);
    }
  }
  return dedup_loads;
}

int64_t ComputeAtReductionTactic::GetSerialLoopExtent(
    const std::vector<ir::Expr>& loops) {
  int64_t extent = 1;
  for (auto& loop : loops) {
    auto* node = loop.As<ir::For>();
    if (node->is_binded()) continue;
    if (!node->extent.is_constant()) return -1;
    extent *= node->extent.as_int64();
  }
  return extent;
}

}  // namespace

std::unique_ptr<ScheduleTactic> CreateComputeAtReductionTactic() {
  return std::make_unique<ComputeAtReductionTactic>();
}

}  // namespace ir
}  // namespace cinn
