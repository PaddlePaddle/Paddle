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
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace ir {

class ComputeAtReductionTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override { return "ComputeAtReductionTactic"; }

 private:
  void ComputeAtReduceInit(ir::IRSchedule* sch, const std::string& block_id);
  void ComputeAtReduceLoad(ir::IRSchedule* sch, const std::string& block_id);

 private:
  ScheduleContext* context_;
  bool compute_at_reduce_init_done_{false};
};

void ComputeAtReductionTactic::Init(ScheduleContext* context) {
  context_ = context;
}

void ComputeAtReductionTactic::Apply(ir::IRSchedule* sch,
                                     const std::string& block_id) {
  const auto ContainsConditionOrLet = [&](const ir::Expr& expr) -> bool {
    const auto condition_or_let = ir::ir_utils::CollectIRNodesWithoutTensor(
        expr, [&](const Expr* x) -> bool {
          if (x->As<ir::IfThenElse>() || x->As<ir::Select>() ||
              x->As<ir::Let>())
            return true;
        });
    return !condition_or_let.empty();
  };
  // Should analyze condition after having dependency tools.
  if (ContainsConditionOrLet(sch->GetRootBlock(sch->GetBlock(block_id))))
    return;

  if (!compute_at_reduce_init_done_) {
    for (const auto& block : sch->GetAllBlocks()) {
      ComputeAtReduceInit(sch,
                          block.As<ir::ScheduleBlockRealize>()
                              ->schedule_block.As<ir::ScheduleBlock>()
                              ->name);
    }
    VLOG(6) << "After ComputeAtReductionInit on expr: [" << block_id
            << "], expr:\n"
            << sch->GetModule().GetExprs().front();
    compute_at_reduce_init_done_ = true;
  }

  ComputeAtReduceLoad(sch, block_id);
  VLOG(6) << "After ComputeAtReductionLoad on expr: [" << block_id
          << "], expr:\n"
          << sch->GetModule().GetExprs().front();
}

bool ForExtentsEqual(const std::vector<ir::Expr>& first,
                     const std::vector<ir::Expr>& second) {
  if (first.size() != second.size()) {
    return false;
  }
  for (size_t i = 0; i < first.size(); ++i) {
    const ir::For* first_for = first[i].As<ir::For>();
    const ir::For* second_for = second[i].As<ir::For>();
    PADDLE_ENFORCE_NOT_NULL(
        first_for,
        ::common::errors::InvalidArgument("The input node should be a For!"));
    PADDLE_ENFORCE_NOT_NULL(
        second_for,
        ::common::errors::InvalidArgument("The input node should be a For!"));

    if (!ir::ir_utils::IRCompare(first_for->extent, second_for->extent)) {
      return false;
    }
    if (first_for->for_type() != second_for->for_type()) {
      return false;
    }
  }
  return true;
}

void ComputeAtReductionTactic::ComputeAtReduceInit(
    ir::IRSchedule* sch, const std::string& block_id) {
  if (!ir::IsReduceInitTensorName(block_id)) return;

  const auto GetRootInitBlockId =
      [&](const std::vector<ir::Expr>& blocks) -> std::string {
    for (const auto& block : blocks) {
      const std::string root_block_name =
          block.As<ir::ScheduleBlockRealize>()
              ->schedule_block.As<ir::ScheduleBlock>()
              ->name;
      if (ir::IsReduceInitTensorName(root_block_name)) return root_block_name;
    }
    return "";
  };

  const std::vector<ir::Expr> blocks = sch->GetAllBlocks();
  const std::string root_init_block_id = GetRootInitBlockId(blocks);
  if (root_init_block_id.empty() || root_init_block_id == block_id) return;

  const std::vector<ir::Expr> root_loops = sch->GetLoops(root_init_block_id);
  const std::vector<ir::Expr> cur_loops = sch->GetLoops(block_id);
  if (!ForExtentsEqual(root_loops, cur_loops)) return;

  sch->SimpleComputeAt(sch->GetBlock(block_id),
                       sch->GetLoops(root_init_block_id).back());
}

std::string FindCandidateBlockId(ir::IRSchedule* sch,
                                 const std::vector<ir::Expr>& blocks,
                                 const ir::Expr& cur_block) {
  const auto ConstructForVarMap = [&](const std::vector<ir::Expr>& lhs_loops,
                                      const std::vector<ir::Expr>& rhs_loops)
      -> std::unordered_map<ir::Var, ir::Var> {
    PADDLE_ENFORCE_EQ(lhs_loops.size(),
                      rhs_loops.size(),
                      ::common::errors::InvalidArgument(
                          "The for loops size should be equal."));
    std::unordered_map<ir::Var, ir::Var> ret;
    for (std::size_t i = 0; i < lhs_loops.size(); ++i) {
      const auto& rhs_var = rhs_loops[i].As<ir::For>()->loop_var;
      ret[lhs_loops[i].As<ir::For>()->loop_var] = rhs_var;
    }
    return ret;
  };

  const auto IndicesWithIterValues =
      [&](const std::vector<ir::Expr>& indices,
          const ir::ScheduleBlockRealize* sbr,
          const std::unordered_map<ir::Var, ir::Var>& for_var_map)
      -> std::vector<ir::Expr> {
    std::vector<ir::Expr> tensor_indices;
    std::vector<ir::Expr> map_iter_values;
    for (const auto& iter_value : sbr->iter_values) {
      ir::Expr map_iter_value = ir::ir_utils::IRCopy(iter_value);
      for (const auto& [lhs_var, rhs_var] : for_var_map) {
        cinn::optim::ReplaceVarWithExpr(
            &map_iter_value, lhs_var, ir::ir_utils::IRCopy(rhs_var));
      }
      map_iter_values.push_back(map_iter_value);
    }
    for (ir::Expr index : indices) {
      ir::Expr index_value = ir::analyzer::ReplaceVarWithExpr(
          index,
          sbr->schedule_block.As<ir::ScheduleBlock>()->iter_vars,
          map_iter_values);
      tensor_indices.push_back(index_value);
    }
    return tensor_indices;
  };

  const auto LoadAndIndicesEqual = [&](const ir::Load* first,
                                       const ir::Load* second,
                                       const ir::Expr& first_block,
                                       const ir::Expr& second_block) -> bool {
    if (first->tensor.as_tensor_ref()->buffer->name !=
        second->tensor.as_tensor_ref()->buffer->name)
      return false;
    const std::vector<ir::Expr> first_loops = sch->GetLoops(first_block);
    const std::vector<ir::Expr> second_loops = sch->GetLoops(second_block);
    if (!ForExtentsEqual(first_loops, second_loops)) return false;
    std::unordered_map<ir::Var, ir::Var> for_var_map =
        ConstructForVarMap(first_loops, second_loops);

    const auto first_indices =
        IndicesWithIterValues(first->indices,
                              first_block.As<ir::ScheduleBlockRealize>(),
                              for_var_map);
    const auto second_indices =
        IndicesWithIterValues(second->indices,
                              second_block.As<ir::ScheduleBlockRealize>(),
                              for_var_map);
    if (first_indices.size() != second_indices.size()) return false;
    for (size_t i = 0; i < first_indices.size(); ++i) {
      if (!ir::ir_utils::IRCompare(first_indices[i], second_indices[i])) {
        VLOG(8) << "Indices not equal, first: " << first_indices[i]
                << " second: " << second_indices[i];
        return false;
      }
    }
    VLOG(8) << "Indices equal";
    return true;
  };

  const auto IndicesContainLoad = [&](const ir::Load* load) -> bool {
    for (const auto& index : load->indices) {
      std::set<Expr> load_tensors = ir::ir_utils::CollectLoadTensors(
          index, /*teller=*/[&](const Expr*) -> bool { return true; });
      if (load_tensors.size() > 0) {
        return true;
      }
    }
    return false;
  };

  const auto HasCommonLoad = [&](const ir::Load* load,
                                 const std::set<ir::Expr>& load_nodes,
                                 const ir::Expr& first_block,
                                 const ir::Expr& second_block) -> bool {
    for (const auto& load_node : load_nodes) {
      PADDLE_ENFORCE_NOT_NULL(load_node.As<ir::Load>(),
                              ::common::errors::InvalidArgument(
                                  "The input node should be a Load!"));
      if (IndicesContainLoad(load_node.As<ir::Load>())) return false;
      if (LoadAndIndicesEqual(
              load, load_node.As<ir::Load>(), first_block, second_block))
        return true;
    }
    return false;
  };

  const auto GetCandidateBlockId =
      [&](const std::vector<ir::Expr>& blocks,
          const ir::Expr& cur_block) -> std::string {
    const auto load_nodes = ir::ir_utils::CollectIRNodesWithoutTensor(
        cur_block, [&](const Expr* x) -> bool {
          return x->As<ir::Load>() &&
                 x->As<ir::Load>()
                         ->tensor.as_tensor_ref()
                         ->buffer->memory_type == ir::MemoryType::Heap;
        });

    std::string candidate_block_id = "";
    for (const auto& block : blocks) {
      const auto common_loads = ir::ir_utils::CollectIRNodesWithoutTensor(
          block, [&](const Expr* x) -> bool {
            return x->As<ir::Load>() &&
                   HasCommonLoad(
                       x->As<ir::Load>(), load_nodes, block, cur_block);
          });
      if (!common_loads.empty()) {
        candidate_block_id = block.As<ir::ScheduleBlockRealize>()
                                 ->schedule_block.As<ir::ScheduleBlock>()
                                 ->name;
        break;
      }
    }
    return candidate_block_id;
  };

  return GetCandidateBlockId(blocks, cur_block);
}

bool IsSafeComputeAt(ir::IRSchedule* sch,
                     const std::string& candidate_block_id,
                     const std::string& block_id) {
  const auto GetLoopsAllSbrs =
      [&](const std::vector<ir::Expr>& loops) -> std::vector<ir::Expr> {
    std::vector<ir::Expr> loop_sbrs;
    if (!loops.empty()) {
      ir::ir_utils::CollectIRNodesWithoutTensor(
          loops[0], [&](const Expr* x) -> bool {
            if (x->As<ir::ScheduleBlockRealize>() &&
                !(ir::IsReduceInitTensorName(
                    x->As<ir::ScheduleBlockRealize>()
                        ->schedule_block.As<ir::ScheduleBlock>()
                        ->name))) {
              loop_sbrs.push_back(*x);
            }
            return false;
          });
    }
    return loop_sbrs;
  };

  const auto GetBlockIndex = [&](const std::vector<ir::Expr>& blocks,
                                 const std::string& block_name) -> size_t {
    for (size_t i = 0; i < blocks.size(); ++i) {
      if (blocks[i]
              .As<ir::ScheduleBlockRealize>()
              ->schedule_block.As<ir::ScheduleBlock>()
              ->name == block_name) {
        return i;
      }
    }
    VLOG(8) << "Can't find block index!";
    return blocks.size();
  };

  const auto GetAffectedSbrs =
      [&](const std::vector<ir::Expr>& blocks,
          const std::string& dst_id,
          const std::string& src_id) -> std::vector<ir::Expr> {
    std::vector<ir::Expr> affected_sbrs;
    size_t dst_index = GetBlockIndex(blocks, dst_id);
    size_t src_index = GetBlockIndex(blocks, src_id);
    PADDLE_ENFORCE_LT(dst_index,
                      src_index,
                      ::common::errors::InvalidArgument(
                          "We should guarantee dst_index < src_index!"));
    PADDLE_ENFORCE_LT(src_index,
                      blocks.size(),
                      ::common::errors::InvalidArgument(
                          "We should guarantee src_index < blocks.size()!"));
    for (size_t i = dst_index; i < src_index; ++i) {
      affected_sbrs.push_back(blocks[i]);
    }
    return affected_sbrs;
  };

  const auto BlockDependency = [&](const ir::Expr& first,
                                   const ir::Expr& second) -> bool {
    if (GetTensor(first)->name == GetTensor(second)->name) return true;
    const auto consumers = GetConsumers(second, sch->GetRootBlock(second));
    const auto producers = GetProducers(second, sch->GetRootBlock(second));
    const auto first_name = first.As<ir::ScheduleBlockRealize>()
                                ->schedule_block.As<ir::ScheduleBlock>()
                                ->name;
    for (const auto& consumer : consumers) {
      if (consumer.As<ir::ScheduleBlockRealize>()
              ->schedule_block.As<ir::ScheduleBlock>()
              ->name == first_name)
        return true;
    }
    for (const auto& producer : producers) {
      if (producer.As<ir::ScheduleBlockRealize>()
              ->schedule_block.As<ir::ScheduleBlock>()
              ->name == first_name)
        return true;
    }
    return false;
  };

  const auto AllBlockSafeComputeAt = [&](const std::vector<ir::Expr>& blocks,
                                         const std::string& dst_id,
                                         const std::string& src_id) -> bool {
    std::vector<ir::Expr> check_blocks = GetLoopsAllSbrs(sch->GetLoops(dst_id));
    const auto src_loop_sbrs = GetLoopsAllSbrs(sch->GetLoops(src_id));
    const auto affected_sbrs = GetAffectedSbrs(blocks, dst_id, src_id);
    check_blocks.insert(
        check_blocks.end(), src_loop_sbrs.begin(), src_loop_sbrs.end());
    check_blocks.insert(
        check_blocks.end(), affected_sbrs.begin(), affected_sbrs.end());
    for (const auto& block : check_blocks) {
      VLOG(8) << "Check dependency: " << block;
      if (src_id == block.As<ir::ScheduleBlockRealize>()
                        ->schedule_block.As<ir::ScheduleBlock>()
                        ->name)
        continue;
      if (BlockDependency(block, sch->GetBlock(src_id))) return false;
    }
    return true;
  };
  return AllBlockSafeComputeAt(
      sch->GetAllBlocks(), candidate_block_id, block_id);
}

void ComputeAtReductionTactic::ComputeAtReduceLoad(
    ir::IRSchedule* sch, const std::string& block_id) {
  // 1. Find candidate block, load buffer with same indices.
  const std::string candidate_block_id =
      FindCandidateBlockId(sch, sch->GetAllBlocks(), sch->GetBlock(block_id));
  if (candidate_block_id.empty() || candidate_block_id == block_id) return;
  VLOG(8) << "Candidate block: " << candidate_block_id;

  // 2. Check condition, loops structure and block-level dependency.
  if (!IsSafeComputeAt(sch, candidate_block_id, block_id)) return;
  VLOG(8) << "Compate at is safe: " << block_id;

  // 3. Compute at schedule.
  const std::vector<ir::Expr> candidate_block_loops =
      sch->GetLoops(candidate_block_id);
  sch->SimpleComputeAt(sch->GetBlock(block_id), candidate_block_loops.back());
}

std::unique_ptr<ScheduleTactic> CreateComputeAtReductionTactic() {
  return std::make_unique<ComputeAtReductionTactic>();
}

}  // namespace ir
}  // namespace cinn
