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

namespace cinn {
namespace ir {

void DynamicShapeGroupScheduler::Schedule() {
  // Fake schedule for test
  int max_spacial_numel = 1;
  ScheduleBlockNode* node = schedule_block_graph_->EndPoints()[0];
  ir::Expr block_realize = node->Block();
  std::vector<ir::Expr> loops = ir_sch_->GetLoops(block_realize);
  ir::Expr extent = loops[0].As<ir::For>()->extent;

  ir::Expr condition1 = ir::LE::Make(extent, Expr(1024));
  std::unique_ptr<ir::IRSchedule> new_ir_sch1 =
      std::make_unique<ir::IRSchedule>(*ir_sch_);
  ScheduleBlockGraph sbg1(*new_ir_sch1);
  sbg1.NodesWalk([&](ir::ScheduleBlockNode* node) {
    new_ir_sch1->Bind(ir_sch_->GetLoops(node->Block())[0], "threadIdx.x");
  });
  ir_schs_.emplace_back(condition1, std::move(new_ir_sch1));

  ir::Expr condition2 = ir::GT::Make(extent, Expr(1024));
  std::unique_ptr<ir::IRSchedule> new_ir_sch2 =
      std::make_unique<ir::IRSchedule>(*ir_sch_);
  ScheduleBlockGraph sbg2(*new_ir_sch2);
  sbg2.NodesWalk([&](ir::ScheduleBlockNode* node) {
    new_ir_sch2->Bind(ir_sch_->GetLoops(node->Block())[0], "threadIdx.x");
  });
  ir_schs_.emplace_back(condition2, std::move(new_ir_sch2));
}

std::vector<std::pair<SymbolicCondition, ir::Expr>>
DynamicShapeGroupScheduler::GetIRs() {
  std::vector<std::pair<SymbolicCondition, ir::Expr>> irs;
  for (auto& sch_pair : ir_schs_) {
    irs.emplace_back(sch_pair.first,
                     sch_pair.second->GetModule().GetExprs()[0]);
  }
  return irs;
}

}  // namespace ir
}  // namespace cinn
