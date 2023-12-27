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

#include "paddle/cinn/ir/group_schedule/dy_shape_group_scheduler.h"
#include "paddle/cinn/ir/group_schedule/tactic/arrange_storage_tactic.h"
#include "paddle/cinn/ir/group_schedule/tactic/compute_inline_tactic.h"

namespace cinn {
namespace ir {

void DynamicShapeGroupScheduler::Init() {
  std::unordered_set<std::string> output_names = OutputTensorNames();
  tactics_.emplace_back(new ComputeInlineTactic(output_names, target_));
  tactics_.emplace_back(new ArrangeStorageTactic(output_names));
}

void DynamicShapeGroupScheduler::Schedule() {
  // Fake schedule for test
  std::vector<Expr> all_blocks = ir_sch_->GetAllBlocks();
  for (int i = 0; i < all_blocks.size(); i++) {
    std::vector<Expr> loops = ir_sch_->GetLoops(all_blocks[i]);
    ir_sch_->Fuse(loops);
  }

  ApplyTactics();
  all_blocks = ir_sch_->GetAllBlocks();
  auto block0_loops = ir_sch_->GetLoops(all_blocks[0]);
  auto splited_loops1 = ir_sch_->Split(block0_loops[0], {1024, -1});

  ir_sch_->Bind(splited_loops1[0], "threadIdx.x");

  ir::Expr predicate1 = ir::LE::Make(Expr(1023), Expr(1024));
  std::unique_ptr<ir::IRSchedule> new_ir_sch1 =
      std::make_unique<ir::IRSchedule>(*ir_sch_);
  ir_schs_.emplace_back(predicate1, std::move(new_ir_sch1));
}

void DynamicShapeGroupScheduler::ApplyTactics() {
  schedule_block_graph_->Update(*ir_sch_);
  for (const auto& tactic : tactics_) {
    auto ApplyTacticFunc = [&](ir::ScheduleBlockNode* node) {
      tactic->Apply(ir_sch_, node->id());
    };
    schedule_block_graph_->DFSTopoWalk(ApplyTacticFunc);
    schedule_block_graph_->Update(*ir_sch_);
  }
}

std::vector<std::pair<SymbolicPredicate, ir::Expr>>
DynamicShapeGroupScheduler::GetIRs() {
  std::vector<std::pair<SymbolicPredicate, ir::Expr>> irs;
  for (auto& sch_pair : ir_schs_) {
    irs.emplace_back(sch_pair.first,
                     sch_pair.second->GetModule().GetExprs()[0]);
  }
  return irs;
}

}  // namespace ir
}  // namespace cinn
