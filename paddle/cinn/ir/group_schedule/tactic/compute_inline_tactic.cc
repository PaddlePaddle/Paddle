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

#include "paddle/cinn/ir/group_schedule/tactic/compute_inline_tactic.h"

#include <string>
#include <vector>

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_inline.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace cinn {
namespace ir {

class ComputeInlineTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override { return "ComputeInlineTactic"; }

 private:
  std::unordered_set<std::string> output_names_;
  cinn::common::Target target_;
};

void ComputeInlineTactic::Init(ScheduleContext* context) {
  output_names_ = context->output_names;
  target_ = context->target;
}

void ComputeInlineTactic::Apply(ir::IRSchedule* sch,
                                const std::string& block_id) {
  // TODO(LiuYang): Compute of ops will be rewritten so that we
  // don't use it in dynamic group_schedule rules temporarily.
  // if (IsProhibitScheduleExternCallBlock(node->Block())) {
  //    return;
  // }
  auto_schedule::AutoInline inliner(target_, output_names_);
  VLOG(6) << "try ComputeInline on: " << block_id
          << ", before ComputeInline, func body: "
          << sch->GetModule().GetExprs().front();
  ir::Expr schedule_block = sch->GetBlock(block_id);
  inliner.Apply(sch, schedule_block);
  VLOG(6) << "try ComputeInline on: " << block_id
          << ", after ComputeInline, func body: "
          << sch->GetModule().GetExprs().front();
}

std::unique_ptr<ScheduleTactic> CreateComputeInlineTactic() {
  return std::make_unique<ComputeInlineTactic>();
}

}  // namespace ir
}  // namespace cinn
