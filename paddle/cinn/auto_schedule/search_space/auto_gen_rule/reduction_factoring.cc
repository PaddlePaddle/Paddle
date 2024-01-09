// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/reduction_factoring.h"

#include <glog/logging.h>

#include "paddle/cinn/auto_schedule/analysis/analyze_ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"

namespace cinn {
namespace auto_schedule {

bool ReductionFactoring::CanApply(const std::string& block_name,
                                  ir::IRSchedule* ir_schedule) const {
  ir::Expr block_expr = ir_schedule->GetBlock(block_name);
  ir::ScheduleBlockRealize* block_realize =
      block_expr.As<ir::ScheduleBlockRealize>();
  CHECK_NOTNULL(block_realize);
  ir::ScheduleBlock* sch_block =
      block_realize->schedule_block.As<ir::ScheduleBlock>();
  CHECK_NOTNULL(sch_block);
  AnalyzeScheduleBlockReadWriteBuffer(sch_block);

  // 1. The block must have write buffer
  if (sch_block->write_buffers.empty()) {
    return false;
  }

  // 2. The block must have at least one reduce axis
  const std::vector<ir::Var>& iter_vars = sch_block->iter_vars;
  bool find_reduce_axis = false;
  for (int i = 0; i < iter_vars.size(); ++i) {
    if (iter_vars[i]->is_reduce_axis) {
      find_reduce_axis = true;
      break;
    }
  }
  if (!find_reduce_axis) {
    return false;
  }

  // 3. Each loop's body only contains one sub loop or block, except reduce_init
  // block
  std::vector<ir::Expr> loops = ir_schedule->GetLoops(block_name);
  for (const ir::Expr& loop : loops) {
    const ir::Expr& body = loop.As<ir::For>()->body;
    if (body.As<ir::Block>()) {
      if (body.As<ir::Block>()->stmts.size() == 1) {
        if (body.As<ir::Block>()->stmts[0].As<ir::For>() == nullptr &&
            body.As<ir::Block>()->stmts[0].As<ir::ScheduleBlockRealize>() ==
                nullptr) {
          return false;
        }
      } else if (body.As<ir::Block>()->stmts.size() == 2) {
        if (body.As<ir::Block>()->stmts[0].As<ir::ScheduleBlockRealize>() ==
                nullptr ||
            !ir::IsReduceInitTensorName(
                GetBlockName(body.As<ir::Block>()->stmts[0]))) {
          return false;
        }
        if (body.As<ir::Block>()->stmts[1].As<ir::For>() == nullptr &&
            body.As<ir::Block>()->stmts[1].As<ir::ScheduleBlockRealize>() ==
                nullptr) {
          return false;
        }
      } else {
        return false;
      }
    } else if (body.As<ir::For>() || body.As<ir::ScheduleBlockRealize>()) {
      continue;
    } else {
      return false;
    }
  }

  return true;
}

RuleApplyType ReductionFactoring::AnalyseApplyType(
    SearchState state, const std::string& block_name) const {
  return this->CanApply(block_name, &(state->ir_schedule))
             ? RuleApplyType::kApply
             : RuleApplyType::kCannotApply;
}

std::vector<SearchState> ReductionFactoring::ApplyOnBlock(
    SearchState state, const std::string& block_name) {
  SearchState new_state = state.Copy();
  Apply(block_name, &(new_state->ir_schedule));
  return {new_state};
}

void ReductionFactoring::Apply(const std::string& block_name,
                               ir::IRSchedule* ir_schedule) {
  ir::Expr block = ir_schedule->GetBlock(block_name);
  std::vector<ir::Expr> all_loops = ir_schedule->GetLoops(block_name);

  std::vector<ir::Expr> new_loop_order;
  size_t num_spatial_loops = 0;
  size_t num_reduction_loops = 0;
  // 1. Add all spatial loops
  std::unordered_set<std::string> reduce_loop_var_names =
      GetReduceLoopVarNames(block);
  for (const ir::Expr& expr : all_loops) {
    if (reduce_loop_var_names.count(expr.As<ir::For>()->loop_var->name) == 0) {
      new_loop_order.push_back(expr);
      ++num_spatial_loops;
    }
  }
  // 2. Add all reduction loops
  for (const ir::Expr& expr : all_loops) {
    if (reduce_loop_var_names.count(expr.As<ir::For>()->loop_var->name) > 0) {
      new_loop_order.push_back(expr);
      ++num_reduction_loops;
    }
  }
  if (num_reduction_loops == 0) {
    return;
  }
  // 3. Reorder if new_loop_order differs from the original order
  CHECK_EQ(all_loops.size(), new_loop_order.size());
  for (int i = 0; i < all_loops.size(); ++i) {
    if (all_loops[i].As<ir::For>()->loop_var->name !=
        new_loop_order[i].As<ir::For>()->loop_var->name) {
      ir_schedule->Reorder(new_loop_order);
      break;
    }
  }

  // 4. Fuse all reduction loops
  ir::Expr fused_reduce_loop;
  VLOG(6) << "before Fuse: " << ir_schedule->GetModule().GetExprs()[0];
  if (num_reduction_loops > 1) {
    std::vector<int> reduction_loop_indices;
    for (int i = num_spatial_loops; i < all_loops.size(); ++i) {
      reduction_loop_indices.push_back(i);
    }
    CHECK_EQ(reduction_loop_indices.size(), num_reduction_loops);
    fused_reduce_loop = ir_schedule->Fuse(block_name, reduction_loop_indices);
  } else {
    all_loops = ir_schedule->GetLoops(block_name);
    fused_reduce_loop = all_loops.back();
  }
  // 5. Split the reduction loop into 2 part
  VLOG(6) << "before Split: " << ir_schedule->GetModule().GetExprs()[0];
  int factor = 1;
  int max_factor = 1024;
  int extent = ir::GetLoopExtent(fused_reduce_loop);
  for (int i = max_factor; i >= 1; --i) {
    if (extent % i == 0) {
      factor = i;
      break;
    }
  }
  std::vector<cinn::ir::Expr> splited_reduction_loops =
      ir_schedule->Split(fused_reduce_loop, {factor, -1});
  // 6.  Apply FactorizeReduction
  VLOG(6) << "before FactorizeReduction: "
          << ir_schedule->GetModule().GetExprs()[0];
  ir_schedule->FactorizeReduction(splited_reduction_loops[0],
                                  num_spatial_loops);
  VLOG(6) << "after FactorizeReduction: "
          << ir_schedule->GetModule().GetExprs()[0];

  // 7. Loop fusion and cross thread reduction
  std::vector<ir::Expr> rb_loops = ir_schedule->GetLoops(block_name);
  ir::Expr rf_block = ir_schedule->GetBlock(block_name + "_rf");
  ir_schedule->SimpleComputeAt(rf_block, rb_loops.back());

  rb_loops = ir_schedule->GetLoops(block_name);
  ir::Expr rf_init_block =
      ir_schedule->GetBlock(block_name + "_rf__reduce_init");
  ir_schedule->SimpleComputeAt(rf_init_block, rb_loops.back());

  if (*target_ == cinn::common::DefaultNVGPUTarget()) {
    rb_loops = ir_schedule->GetLoops(block_name);
    rf_block = ir_schedule->GetBlock(block_name + "_rf");
    ir_schedule->Bind(rb_loops.back(), "threadIdx.x");
    ir_schedule->SetBuffer(rf_block, "shared");
  }
  VLOG(6) << "Loop fusion and cross thread reduction: "
          << ir_schedule->GetModule().GetExprs()[0];
}

}  // namespace auto_schedule
}  // namespace cinn
