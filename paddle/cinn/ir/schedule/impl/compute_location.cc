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

#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/ir/schedule/impl/ir_schedule.h"
#include "paddle/common/enforce.h"
/** \brief A macro that guards the beginning of each implementation of schedule
 */
#define CINN_IR_SCHEDULE_BEGIN() try {
/**
 * \brief A macro that pairs with `CINN_IR_SCHEDULE_BEGIN`, handling potential
 * errors and error message printing.
 * @param primitive A string representing the kind of schedule primitive.
 * @param err_msg_level A ScheduleErrorMessageLevel enum, level of error message
 * printing
 */
#define CINN_IR_SCHEDULE_END(err_msg_level)              \
  }                                                      \
  catch (const utils::ErrorHandler& err_handler) {       \
    PADDLE_THROW(::common::errors::Fatal(                \
        err_handler.FormatErrorMessage(err_msg_level))); \
  }

namespace cinn {
namespace ir {

void DyScheduleImpl::ComputeAt(const Expr& block,
                               const Expr& loop,
                               bool keep_unit_loops) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "ComputeAt";
  std::ostringstream os;
  PADDLE_ENFORCE_NOT_NULL(
      block.As<ir::ScheduleBlockRealize>(),
      ::common::errors::InvalidArgument(
          "[IRScheduleError] An error occurred in the schedule primitive "
          "<ComputeAt>.\n"
          "[Error info] Expr param(block) should be a ScheduleBlockRealize!\n"
          "[Expr info] The Expr of current schedule is: %s.",
          module_expr_.GetExprs()));
  PADDLE_ENFORCE_NOT_NULL(
      loop.As<ir::For>(),
      ::common::errors::InvalidArgument(
          "[IRScheduleError] An error occurred in the schedule primitive "
          "<ComputeAt>.\n"
          "[Error info] Expr param(loop) should be a For node!\n"
          "[Expr info] The Expr of current schedule is: %s.",
          module_expr_.GetExprs()));
  Expr root = this->GetRootBlock(block);

  VLOG(3) << "Begin ComputeAt of loop:\n" << loop << "\nat block:\n" << root;

  auto producers = GetProducers(block, root);
  auto consumers = GetConsumers(block, root);
  CheckComputeAtValidation(block, loop, root);
  LoopReconstructor reconstructor(root, block, loop);
  LeafBlockRemovalPlan remove_plan(
      block, &reconstructor.source_expr, &reconstructor.target_expr);
  remove_plan(&root);
  auto iter_ranges = CalculateRequiredRegions(block, loop, root, consumers);
  std::string new_var_names =
      reconstructor.MakeNewLoop(iter_ranges, keep_unit_loops, 0);
  auto sch_block_expr = block.As<ir::ScheduleBlockRealize>()->schedule_block;
  sch_block_expr.As<ir::ScheduleBlock>()->attrs.emplace(
      ir::attr::compute_at_extra_var, new_var_names);
  this->Replace(reconstructor.source_expr, reconstructor.target_expr);
  this->Replace(reconstructor.loop_, reconstructor.new_loop_);

  VLOG(3) << "After ComputeAt, ir is:\n" << reconstructor.new_loop_;
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::SimpleComputeAt(const Expr& block, const Expr& loop) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "SimpleComputeAt";
  std::ostringstream os;
  PADDLE_ENFORCE_NOT_NULL(
      block.As<ScheduleBlockRealize>(),
      ::common::errors::InvalidArgument(
          "[IRScheduleError] An error occurred in the schedule primitive "
          "<SimpleComputeAt>.\n"
          "[Error info] Expr param(block) should be a ScheduleBlockRealize!\n"
          "[Expr info] The Expr of current schedule is: %s.",
          module_expr_.GetExprs()));
  PADDLE_ENFORCE_NOT_NULL(
      loop.As<For>(),
      ::common::errors::InvalidArgument(
          "[IRScheduleError] An error occurred in the schedule primitive "
          "<SimpleComputeAt>.\n"
          "[Error info] Expr param(loop) should be a For node!\n"
          "[Expr info] The Expr of current schedule is: %s.",
          module_expr_.GetExprs()));
  std::vector<Expr> block_loops = this->GetLoops(block);
  Expr root = this->GetRootBlock(block);
  auto loops = GetLoopsOfExpr(loop, root);

  VLOG(3) << "Begin SimpleComputeAt of loop:\n"
          << loop << "\nat block:\n"
          << root;

  auto this_loop = loop;
  auto block_name = GetTensor(block)->name;
  auto this_block = block;
  if (loops[0].As<ir::For>()->extent.is_constant() &&
      GetLoopExtent(loops[0]) == 1 &&
      (!block_loops[0].As<ir::For>()->extent.is_constant() ||
       GetLoopExtent(block_loops[0]) != 1)) {
    this->Split(block_loops[0], {1, -1});
    this_block = this->GetBlock(block_name);
  } else if ((!loops[0].As<ir::For>()->extent.is_constant() ||
              GetLoopExtent(loops[0]) != 1) &&
             block_loops[0].As<ir::For>()->extent.is_constant() &&
             GetLoopExtent(block_loops[0]) == 1) {
    auto splited = this->Split(loops[0], {1, -1});
    this_loop = splited[1];
  }

  block_loops = this->GetLoops(this_block);
  root = this->GetRootBlock(this_block);
  loops = GetLoopsOfExpr(this_loop, root);

  PADDLE_ENFORCE_LE(loops.size(),
                    block_loops.size(),
                    ::common::errors::InvalidArgument(
                        "The size of loops should be less than or "
                        "equal to the size of block_loops."));

  std::vector<Var> replaced_var;
  std::vector<Expr> substitute_expr;
  common::cas_intervals_t var_intervals;
  common::SymbolicExprAnalyzer analyzer{var_intervals};
  for (int i = 0; i < loops.size(); ++i) {
    VLOG(3) << i << "-th loop is:\n " << loops[i];
    VLOG(3) << i << "-th block_loop:\n" << block_loops[i];
    std::optional<bool> prove_eq = analyzer.ProveEQ(
        loops[i].As<ir::For>()->extent, block_loops[i].As<ir::For>()->extent);
    PADDLE_ENFORCE_EQ(
        prove_eq.has_value(),
        true,
        ::common::errors::InvalidArgument(
            "[IRScheduleError] An error occurred in the schedule primitive "
            "<SimpleComputeAt>.\n"
            "[Error info] Extent of loop in Expr Param(loop) and extent of "
            "loop in Expr Param(block) should be equal correspondingly!\n"
            "[Expr info] The Expr of current schedule is: %s.",
            module_expr_.GetExprs()));

    PADDLE_ENFORCE_EQ(
        prove_eq.value(),
        true,
        ::common::errors::InvalidArgument(
            "[IRScheduleError] An error occurred in the schedule primitive "
            "<SimpleComputeAt>.\n"
            "[Error info] Extent of loop in Expr Param(loop) and extent of "
            "loop in Expr Param(block) should be equal correspondingly!\n"
            "[Expr info] The Expr of current schedule is: %s.",
            module_expr_.GetExprs()));
    if (block_loops[i].As<ir::For>()->bind_info().valid() &&
        !loops[i].As<ir::For>()->bind_info().valid()) {
      loops[i].As<ir::For>()->set_bind_info(
          block_loops[i].As<ir::For>()->bind_info());
    }
    replaced_var.push_back(block_loops[i].As<ir::For>()->loop_var);
    substitute_expr.push_back(Expr(loops[i].As<ir::For>()->loop_var));
  }

  Expr result = loops.size() < block_loops.size()
                    ? ir::ir_utils::IRCopy(block_loops[loops.size()])
                    : ir::ir_utils::IRCopy(this_block);
  Expr new_loop = ir::ir_utils::IRCopy(this_loop);

  // Get the body of block_loop under the same loops
  auto body = block_loops.at(loops.size() - 1).As<ir::For>()->body;
  // collect if
  auto if_checker = [](const Expr* x) { return x->As<ir::IfThenElse>(); };
  auto if_set = ir::ir_utils::CollectIRNodesWithoutTensor(body, if_checker);
  auto checker = [block_name](const Expr* x) {
    return x->As<ir::ScheduleBlockRealize>() &&
           x->As<ir::ScheduleBlockRealize>()
                   ->schedule_block.As<ScheduleBlock>()
                   ->name == block_name;
  };
  for (auto if_expr : if_set) {
    if (Contains(result, if_expr)) continue;
    if (ir::ir_utils::CollectIRNodesWithoutTensor(if_expr, checker, true)
            .size() > 0) {
      result = IfThenElse::Make(
          ir::ir_utils::IRCopy(if_expr.As<ir::IfThenElse>()->condition),
          result);
      break;
    }
  }

  ReplaceExpr(&result, replaced_var, substitute_expr);
  // When there are two identical IfThenElse
  if (new_loop.As<ir::For>() && new_loop.As<ir::For>()->body.As<ir::Block>() &&
      new_loop.As<ir::For>()
          ->body.As<ir::Block>()
          ->stmts[0]
          .As<ir::IfThenElse>()) {
    auto if_then_else = new_loop.As<ir::For>()->body.As<ir::Block>()->stmts[0];
    if (result.As<ir::IfThenElse>() &&
        if_then_else.As<ir::IfThenElse>()->condition ==
            result.As<ir::IfThenElse>()->condition) {
      new_loop.As<ir::For>()
          ->body.As<ir::Block>()
          ->stmts[0]
          .As<ir::IfThenElse>()
          ->true_case = ir::Block::Make({result.As<ir::IfThenElse>()->true_case,
                                         new_loop.As<ir::For>()
                                             ->body.As<ir::Block>()
                                             ->stmts[0]
                                             .As<ir::IfThenElse>()
                                             ->true_case});
    } else {
      std::vector<ir::Expr>::iterator pos =
          new_loop.As<ir::For>()->body.As<ir::Block>()->stmts.begin();
      new_loop.As<ir::For>()->body.As<ir::Block>()->stmts.insert(pos, result);
    }
  } else {
    if (new_loop.As<ir::For>()->body.As<ir::Block>()) {
      std::vector<ir::Expr> new_body{result};
      for (const auto& stmt :
           new_loop.As<ir::For>()->body.As<ir::Block>()->stmts) {
        new_body.push_back(stmt);
      }
      new_loop.As<ir::For>()->body = ir::Block::Make(new_body);
    } else {
      new_loop.As<ir::For>()->body =
          ir::Block::Make({result, new_loop.As<ir::For>()->body});
    }
  }

  Expr source_expr{nullptr};
  Expr target_expr{nullptr};

  LeafBlockRemovalPlan remove_plan(
      result.As<ir::For>() ? block_loops[loops.size()] : this_block,
      &source_expr,
      &target_expr);
  remove_plan(&root);

  this->Replace(source_expr, target_expr);
  this->Replace(this_loop, new_loop);

  VLOG(3) << "After SimpleComputeAt, ir is:\n" << new_loop;
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::ReverseComputeAt(const Expr& block,
                                      const Expr& loop,
                                      bool keep_unit_loops) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "ReverseComputeAt";
  std::ostringstream os;
  PADDLE_ENFORCE_NOT_NULL(
      block.As<ir::ScheduleBlockRealize>(),
      ::common::errors::InvalidArgument(
          "The block argument must be of type ScheduleBlockRealize."));

  PADDLE_ENFORCE_NOT_NULL(loop.As<ir::For>(),
                          ::common::errors::InvalidArgument(
                              "The loop argument must be of type For."));
  Expr root = this->GetRootBlock(block);
  auto producers = GetProducers(block, root);
  auto consumers = GetConsumers(block, root);
  CheckComputeAtValidation(block, loop, root);
  LoopReconstructor reconstructor(root, block, loop);
  LeafBlockRemovalPlan remove_plan(
      block, &reconstructor.source_expr, &reconstructor.target_expr);
  remove_plan(&root);
  auto iter_ranges =
      CalculateRequiredRegions(block, loop, root, producers, false);
  std::string new_var_names =
      reconstructor.MakeNewLoop(iter_ranges, keep_unit_loops, -1);
  auto sch_block_expr = block.As<ir::ScheduleBlockRealize>()->schedule_block;
  sch_block_expr.As<ir::ScheduleBlock>()->attrs.emplace(
      ir::attr::reverse_compute_at_extra_var, new_var_names);
  this->Replace(reconstructor.source_expr, reconstructor.target_expr);
  this->Replace(reconstructor.loop_, reconstructor.new_loop_);
  return;
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::ComputeInline(const Expr& schedule_block) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "ComputeInline";
  std::ostringstream os;

  PADDLE_ENFORCE_NOT_NULL(
      schedule_block.As<ir::ScheduleBlockRealize>(),
      ::common::errors::InvalidArgument(
          "[IRScheduleError] An error occurred in the schedule primitive "
          "<ComputeInline>.\n"
          "[Error info] Expr param(schedule_block) should be a "
          "ScheduleBlockRealize!\n"
          "[Expr info] The Expr of current schedule is: %s.",
          module_expr_.GetExprs()));

  Expr root = this->GetRootBlock(schedule_block);
  Expr store = CheckComputeInlineValidationAndGetStore(schedule_block, root);
  ComputeInliner inliner(store.As<ir::Store>()->tensor.as_tensor_ref(), store);

  PADDLE_ENFORCE_EQ(
      inliner.BodyPatternAllowInline(),
      true,
      ::common::errors::InvalidArgument(
          "[IRScheduleError] An error occurred in the schedule primitive "
          "<ComputeInline>.\n"
          "[Error info] Current IR can't meets the requirements of "
          "ComputeInline!\n"
          "[Expr info] The Expr of current schedule is: %s.",
          module_expr_.GetExprs()));

  // Create a plan that removes the block to be inlined
  LeafBlockRemovalPlan remove_plan(
      schedule_block, &inliner.src_stmt, &inliner.tgt_stmt);
  remove_plan(&root);
  inliner(&root);
  return;
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::ReverseComputeInline(const Expr& schedule_block) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "ReverseComputeInline";
  std::ostringstream os;
  Expr root = this->GetRootBlock(schedule_block);
  auto exprs =
      CheckReverseComputeInlineValidationAndGetExprs(schedule_block, root);
  Expr inlined_load = std::get<0>(exprs);
  Expr inlined_store = std::get<1>(exprs);
  Expr target_store = std::get<2>(exprs);
  ReverseComputeInliner inliner(
      inlined_store.As<ir::Store>()->tensor.as_tensor_ref(),
      inlined_store,
      inlined_load,
      target_store);

  PADDLE_ENFORCE_EQ(
      inliner.BodyPatternAllowInline(),
      true,
      ::common::errors::InvalidArgument(
          "[IRScheduleError] An error occurred in the schedule primitive "
          "<ReverseComputeInline>.\n"
          "[Error info] Current IR can't meets the requirements of "
          "ReverseComputeInline!\n"
          "[Expr info] The Expr of current schedule is: %s.",
          module_expr_.GetExprs()));
  // Create a plan that removes the block to be inlined
  LeafBlockRemovalPlan remove_plan(
      schedule_block, &inliner.src_stmt, &inliner.tgt_stmt);
  remove_plan(&root);
  inliner(&root);
  inliner(&root);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

}  // namespace ir
}  // namespace cinn
