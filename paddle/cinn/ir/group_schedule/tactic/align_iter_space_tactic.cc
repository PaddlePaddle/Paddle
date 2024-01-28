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

#include "paddle/cinn/ir/group_schedule/tactic/align_iter_space_tactic.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_copy.h"

namespace cinn {
namespace ir {

void AlignIterSpaceTactic::Init(ScheduleContext* context) {
  context_ = context;
}

void AlignIterSpaceTactic::Apply(ir::IRSchedule* sch,
                                 const std::string& block_id) {
  ir::Expr block = sch->GetBlock(block_id);
  if (analyzer::IsReductionSBlock(block)) {
    return;
  }

  std::vector<ir::Expr> loops = sch->GetLoops(block_id);
  ir::Expr src_fused_loop = sch->Fuse(loops);
  ir::Expr src_total_extent = src_fused_loop.As<ir::For>()->extent;

  ir::Expr target_sp_extent{1};
  for (const auto& iter : context_->iter_space_info.sp_space) {
    target_sp_extent = target_sp_extent * std::get<0>(iter);
  }
  ir::Expr target_total_extent = ir_utils::IRCopy(target_sp_extent);
  for (const auto& iter : context_->iter_space_info.rb_space) {
    target_total_extent = target_total_extent * std::get<0>(iter);
  }

  common::cas_intervals_t var_intervals;
  common::SymbolicExprAnalyzer symbolic_expr_analyzer(var_intervals);
  std::optional<bool> total_extent_eq =
      symbolic_expr_analyzer.ProveEQ(src_total_extent, target_total_extent);
  bool need_reorder = false;
  for (int i = 0; i < context_->iter_space_info.rb_last_order.size(); ++i) {
    if (context_->iter_space_info.rb_last_order[i] != i) {
      need_reorder = true;
      break;
    }
  }

  if (total_extent_eq.has_value() && total_extent_eq.value()) {
    sch->Split(src_fused_loop,
               context_->iter_space_info.memory_consistent_order_space);
    loops = sch->GetLoops(block_id);
    if (need_reorder) {
      sch->Reorder(block_id, context_->iter_space_info.rb_last_order);
    }
    if (context_->iter_space_info.sp_space.size() < loops.size() - 1) {
      loops = sch->GetLoops(block_id);
      std::vector<ir::Expr> rb_loops(
          loops.begin() + context_->iter_space_info.sp_space.size(),
          loops.end());
      sch->Fuse(rb_loops);
    }
    if (context_->iter_space_info.sp_space.size() > 1) {
      loops = sch->GetLoops(block_id);
      std::vector<ir::Expr> sp_loops(
          loops.begin(),
          loops.begin() + context_->iter_space_info.sp_space.size());
      sch->Fuse(sp_loops);
    }
  }
}

}  // namespace ir
}  // namespace cinn
