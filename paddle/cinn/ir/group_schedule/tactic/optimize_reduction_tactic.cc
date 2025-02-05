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

#include "paddle/cinn/ir/group_schedule/tactic/optimize_reduction_tactic.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace ir {

class OptimizeReductionTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override { return "OptimizeReductionTactic"; }

 private:
  ScheduleContext* context_;
};

void OptimizeReductionTactic::Init(ScheduleContext* context) {
  context_ = context;
}

bool CanApply(const std::string& block_name, ir::IRSchedule* sch) {
  ir::Expr block_expr = sch->GetBlock(block_name);
  ir::ScheduleBlockRealize* block_realize =
      block_expr.As<ir::ScheduleBlockRealize>();
  PADDLE_ENFORCE_NOT_NULL(block_realize,
                          ::common::errors::InvalidArgument(
                              "The block is not a ScheduleBlockRealize"));
  ir::ScheduleBlock* sch_block =
      block_realize->schedule_block.As<ir::ScheduleBlock>();
  PADDLE_ENFORCE_NOT_NULL(
      sch_block,
      ::common::errors::InvalidArgument("The block is not a ScheduleBlock"));
  analyzer::AnalyzeScheduleBlockReadWriteBuffer(sch_block);

  // 1. The block must have write buffer
  if (sch_block->write_buffers.empty()) {
    VLOG(6) << "the block: " << block_name
            << " do not have write buffer, so can not apply "
               "OptimizeReductionTactic";
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
    VLOG(6)
        << "the block: " << block_name
        << " do not have reduce axis, so can not apply OptimizeReductionTactic";
    return false;
  }

  // 3. Each loop's body only contains one sub loop or block, except reduce_init
  // block
  std::vector<ir::Expr> loops = sch->GetLoops(block_name);
  for (const ir::Expr& loop : loops) {
    const ir::Expr& body = loop.As<ir::For>()->body;
    if (body.As<ir::Block>()) {
      if (body.As<ir::Block>()->stmts.size() == 1) {
        if (body.As<ir::Block>()->stmts[0].As<ir::For>() == nullptr &&
            body.As<ir::Block>()->stmts[0].As<ir::ScheduleBlockRealize>() ==
                nullptr &&
            body.As<ir::Block>()->stmts[0].As<ir::IfThenElse>() == nullptr) {
          VLOG(6) << "the block: " << block_name
                  << " has a block stmt that is not any of "
                     "schedule_block/for_loop/if, so can not apply "
                     "OptimizeReductionTactic";
          return false;
        }
      } else if (body.As<ir::Block>()->stmts.size() == 2) {
        if (body.As<ir::Block>()->stmts[0].As<ir::ScheduleBlockRealize>() ==
                nullptr ||
            !ir::IsReduceInitTensorName(
                analyzer::GetBlockName(body.As<ir::Block>()->stmts[0]))) {
          return false;
        }
        if (body.As<ir::Block>()->stmts[1].As<ir::For>() == nullptr &&
            body.As<ir::Block>()->stmts[1].As<ir::ScheduleBlockRealize>() ==
                nullptr &&
            body.As<ir::Block>()->stmts[0].As<ir::IfThenElse>() == nullptr) {
          VLOG(6) << "the block: " << block_name
                  << " has a block stmt that is not any of "
                     "schedule_block/for_loop/if, so can not apply "
                     "OptimizeReductionTactic";
          return false;
        }
      } else {
        VLOG(6) << "the block: " << block_name
                << " contains more than 2 statements, so can not apply "
                   "OptimizeReductionTactic";
        return false;
      }
    } else if (body.As<ir::For>() || body.As<ir::ScheduleBlockRealize>() ||
               body.As<ir::IfThenElse>()) {
      continue;
    } else {
      VLOG(6)
          << "the block: " << block_name
          << " has a loop body that is not any of schedule_block/for_loop/if, "
             "so can not apply OptimizeReductionTactic";
      return false;
    }
  }

  return true;
}

void OptimizeReductionTactic::Apply(ir::IRSchedule* sch,
                                    const std::string& block_id) {
  if (!CanApply(block_id, sch)) return;

  std::vector<ir::Expr> loops = sch->GetLoops(block_id);
  int first_reduce_loop_idx = context_->iter_space_info.sp_space.size();
  PADDLE_ENFORCE_LT(
      first_reduce_loop_idx,
      loops.size(),
      ::common::errors::InvalidArgument(
          "first_reduce_loop_idx should be less than number of loop."));
  ir::Expr block = sch->GetBlock(block_id);
  ir::Tensor reduce_tensor = analyzer::GetStoreTensorOfSBlock(block);
  int non_reduce_memory_space_rank =
      reduce_tensor->domain_without_reduce_axis().size();
  // Apply FactorizeReduction
  VLOG(6) << "before FactorizeReduction: " << sch->GetModule().GetExprs()[0];
  sch->FactorizeReduction(loops[first_reduce_loop_idx],
                          non_reduce_memory_space_rank);
  VLOG(6) << "after FactorizeReduction: " << sch->GetModule().GetExprs()[0];

  // Loop fusion and cross thread reduction
  std::vector<ir::Expr> rb_loops = sch->GetLoops(block_id);
  std::string rf_block_id = block_id + "_rf";
  ir::Expr rf_block = sch->GetBlock(rf_block_id);
  sch->SimpleComputeAt(rf_block, rb_loops.back());

  rb_loops = sch->GetLoops(block_id);
  ir::Expr rf_init_block =
      sch->GetBlock(ir::GenReduceInitTensorNameOf(rf_block_id));
  sch->SimpleComputeAt(rf_init_block, rb_loops.back());

  context_->target.arch.Match(
      [&](common::NVGPUArch) {
        rb_loops = sch->GetLoops(block_id);
        rf_block = sch->GetBlock(rf_block_id);
        sch->Bind(rb_loops.back(), "threadIdx.x");
        sch->SetBuffer(rf_block, "local");
      },
      [&](std::variant<common::UnknownArch, common::X86Arch, common::ARMArch>) {
      },
      [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
        rb_loops = sch->GetLoops(block_id);
        rf_block = sch->GetBlock(rf_block_id);
        sch->Bind(rb_loops.back(), "threadIdx.x");
        sch->SetBuffer(rf_block, "local");
      });

  VLOG(6) << "Loop fusion and cross thread reduction: "
          << sch->GetModule().GetExprs()[0];
}

std::unique_ptr<ScheduleTactic> CreateOptimizeReductionTactic() {
  return std::make_unique<OptimizeReductionTactic>();
}

}  // namespace ir
}  // namespace cinn
