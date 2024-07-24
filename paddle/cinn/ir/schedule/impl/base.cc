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

#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
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
    PADDLE_THROW(phi::errors::InvalidArgument(           \
        err_handler.FormatErrorMessage(err_msg_level))); \
  }

namespace cinn {
namespace ir {

void DyScheduleImpl::MergeExprs() {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "MergeExprs";
  std::ostringstream os;
  auto exprs = this->GetModule().GetExprs();
  if (exprs.size() <= 1U) return;
  if (!exprs[0].As<ir::Block>()) {
    os << "Expr[0] of module_expr should be a Block!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }
  if (exprs[0].As<ir::Block>()->stmts.size() != 1U) {
    os << "Expr[0] of module_expr should have only one stmt!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }
  if (!exprs[0].As<ir::Block>()->stmts[0].As<ir::ScheduleBlockRealize>()) {
    os << "Expr[0] of module_expr should be Block with only one stmt which is "
          "a "
          "ScheduleBlockRealize!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }
  if (!exprs[0]
           .As<ir::Block>()
           ->stmts[0]
           .As<ir::ScheduleBlockRealize>()
           ->schedule_block.As<ir::ScheduleBlock>()) {
    os << "Expr[0] of module_expr should be Block with only one stmt which is "
          "a "
          "ScheduleBlockRealize with a defined ScheduleBlock!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  std::vector<Expr> merged_block;
  merged_block.push_back(exprs[0]
                             .As<ir::Block>()
                             ->stmts[0]
                             .As<ir::ScheduleBlockRealize>()
                             ->schedule_block.As<ir::ScheduleBlock>()
                             ->body);
  VLOG(3) << "Before merging, exprs[0] is : " << exprs[0];
  for (int i = 1; i < exprs.size(); ++i) {
    auto root_block = ir::ir_utils::CollectIRNodesWithoutTensor(
        exprs[i],
        [&](const Expr* x) {
          return x->As<ir::ScheduleBlockRealize>() &&
                 x->As<ir::ScheduleBlockRealize>()->iter_values.empty();
        },
        true);
    PADDLE_ENFORCE_EQ(
        root_block.size(),
        1U,
        phi::errors::InvalidArgument("Number of root block should be 1"));
    for (auto& it_block : root_block) {
      auto& block_body = it_block.As<ir::ScheduleBlockRealize>()
                             ->schedule_block.As<ir::ScheduleBlock>()
                             ->body;
      merged_block.push_back(block_body);
    }
  }
  for (auto& block : merged_block) {
    VLOG(3) << "in merged_block, it has \n" << block;
  }
  auto merged_expr = ir::Block::Make(merged_block);
  exprs[0]
      .As<ir::Block>()
      ->stmts[0]
      .As<ir::ScheduleBlockRealize>()
      ->schedule_block.As<ir::ScheduleBlock>()
      ->body = merged_expr;
  VLOG(3) << "After merging, exprs[0] is : " << exprs[0];
  exprs.erase(exprs.begin() + 1, exprs.end());
  this->SetExprs(exprs);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

bool DyScheduleImpl::HasBlock(const std::string& block_name) const {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "HasBlock";
  std::ostringstream os;
  auto exprs = module_expr_.GetExprs();
  return analyzer::HasBlock(exprs, block_name);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

std::vector<Expr> DyScheduleImpl::GetLoops(const Expr& block) const {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "GetLoops";
  std::ostringstream os;
  auto exprs = module_expr_.GetExprs();
  return analyzer::GetLoops(exprs, block);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

std::vector<Expr> DyScheduleImpl::GetLoops(
    const std::string& block_name) const {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "GetLoops";
  std::ostringstream os;
  auto exprs = module_expr_.GetExprs();
  return analyzer::GetLoops(exprs, block_name);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

std::vector<Expr> DyScheduleImpl::GetAllBlocks() const {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "GetAllBlocks";
  std::ostringstream os;
  auto exprs = module_expr_.GetExprs();
  return analyzer::GetAllBlocks(exprs);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

std::vector<Expr> DyScheduleImpl::GetChildBlocks(const Expr& expr) const {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "GetChildBlocks";
  std::ostringstream os;
  return analyzer::GetChildBlocks(expr);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

Expr DyScheduleImpl::GetBlock(const std::string& block_name) const {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "GetBlock";
  std::ostringstream os;
  auto exprs = module_expr_.GetExprs();
  return analyzer::GetBlock(exprs, block_name);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

Expr DyScheduleImpl::GetRootBlock(const Expr& expr) const {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "GetRootBlock";
  std::ostringstream os;
  auto exprs = module_expr_.GetExprs();
  return analyzer::GetRootBlock(exprs, expr);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

DeviceAPI DyScheduleImpl::GetDeviceAPI() const {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "GetDeviceAPI";
  std::ostringstream os;
  auto exprs = module_expr_.GetExprs();
  return analyzer::GetDeviceAPI(exprs);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::Annotate(const Expr& block,
                              const std::string& key,
                              const attr_t& value) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "Annotate";
  std::ostringstream os;
  if (!block.As<ir::ScheduleBlockRealize>()) {
    os << "Expr param(block) must be a ScheduleBlockRealize!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }
  if (!block.As<ir::ScheduleBlockRealize>()
           ->schedule_block.As<ScheduleBlock>()) {
    os << "Expr param(block) must be a ScheduleBlockRealize with a "
          "defined ScheduleBlock!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  auto copied_block = ir::ir_utils::IRCopy(block);
  auto* schedule_block = copied_block.As<ir::ScheduleBlockRealize>()
                             ->schedule_block.As<ir::ScheduleBlock>();
  schedule_block->attrs.emplace(key, value);
  this->Replace(block, copied_block);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::Unannotate(Expr& block,
                                const std::string& ann_key) {  // NOLINT
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "Unannotate";
  std::ostringstream os;
  if (!block.As<ir::ScheduleBlockRealize>()) {
    os << "Expr param(block) must be a ScheduleBlockRealize!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }
  if (!block.As<ir::ScheduleBlockRealize>()
           ->schedule_block.As<ScheduleBlock>()) {
    os << "Expr param(block) must be a ScheduleBlockRealize with "
          "a defined ScheduleBlock!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  auto* schedule_block = block.As<ir::ScheduleBlockRealize>()
                             ->schedule_block.As<ir::ScheduleBlock>();
  if (schedule_block->attrs.count(ann_key)) {
    schedule_block->attrs.erase(ann_key);
  } else {
    LOG(WARNING) << "Can't find annotation with key: " << ann_key;
    return;
  }
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::CopyTransformAndLoopInfo(const Expr& block,
                                              const Expr& block_target) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "CopyTransformAndLoopInfo";
  std::ostringstream os;

  if (!block.As<ir::ScheduleBlockRealize>()) {
    os << "Expr param(block) must be a "
          "ScheduleBlockRealize!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }
  if (!block_target.As<ir::ScheduleBlockRealize>()) {
    os << "Expr param(block_target) must be a "
          "ScheduleBlockRealize!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }
  auto exprs = this->GetModule().GetExprs();
  if (exprs.size() != 1U) {
    os << "Size of exprs of current module must be 1!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  auto expr = exprs[0];
  auto vars = block.As<ir::ScheduleBlockRealize>()
                  ->schedule_block.As<ir::ScheduleBlock>()
                  ->iter_vars;
  auto vars_target = block_target.As<ir::ScheduleBlockRealize>()
                         ->schedule_block.As<ir::ScheduleBlock>()
                         ->iter_vars;
  auto old_iter_values = block.As<ir::ScheduleBlockRealize>()->iter_values;
  auto iter_values_target =
      block_target.As<ir::ScheduleBlockRealize>()->iter_values;
  std::vector<Expr> new_iter_values;
  for (int i = 0; i < vars.size() && i < vars_target.size(); ++i) {
    if (!(vars[i]->upper_bound.defined() &&
          vars_target[i]->upper_bound.defined())) {
      os << "Upper bound of iter_vars in both Expr param(block) and Expr "
            "param(block_target) must be defined!\n";
      throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
    }
    if (vars[i]->upper_bound.is_constant() &&
        vars_target[i]->upper_bound.is_constant() &&
        vars[i]->upper_bound.get_constant() ==
            vars_target[i]->upper_bound.get_constant() &&
        !vars[i]->is_reduce_axis && !vars_target[i]->is_reduce_axis) {
      new_iter_values.push_back(iter_values_target[i]);
      VLOG(3) << "new_iter_values.push_back " << iter_values_target[i];
    } else {
      break;
    }
  }

  if (new_iter_values.empty()) {
    os << "Cannot CopyTransformAndLoopInfo since shape[0] of source "
          "and target is not equal! "
       << vars[0]->upper_bound << " v.s " << vars_target[0]->upper_bound;
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  int changed_loop_num = new_iter_values.size();
  std::set<std::string> used_target_loop_vars;
  for (auto& iter_val : new_iter_values) {
    auto find_partial_loop =
        ir::ir_utils::CollectIRNodesWithoutTensor(iter_val, [&](const Expr* x) {
          if (x->as_var()) used_target_loop_vars.insert(x->as_var_ref()->name);
          return x->as_var();
        });
  }
  if (used_target_loop_vars.empty()) {
    os << "Cannot CopyTransformAndLoopInfo since there is no loop var in the "
          "new_iter_values!";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  std::vector<Expr> used_target_loops;
  auto expr_copy = ir::ir_utils::IRCopy(expr);
  for (auto& var : used_target_loop_vars) {
    auto find_loop_var = ir::ir_utils::CollectIRNodesWithoutTensor(
        expr_copy,
        [&](const Expr* x) {
          return x->As<ir::For>() && x->As<ir::For>()->loop_var->name == var &&
                 Contains(*x, block_target);
        },
        true);
    if (find_loop_var.size() != 1U) {
      os << "Number of loop with iter_var which is used in "
            "ScheduleBlockRealize for indexing in Exprs[0] of module_exprs "
            "must be 1!\n";
      throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
    }
    used_target_loops.push_back(*find_loop_var.begin());
    VLOG(3) << "used_target_loops push_back " << used_target_loops.back();
  }
  std::sort(
      used_target_loops.begin(), used_target_loops.end(), [&](Expr i, Expr j) {
        return (utils::GetStreamCnt(i).size() > utils::GetStreamCnt(j).size());
      });
  for (int i = new_iter_values.size(); i < old_iter_values.size(); ++i) {
    if (!old_iter_values[i].as_var()) {
      os << "iter_vars[" << i << "] in Expr param(block) must be vars!\n";
      throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
    }
    new_iter_values.push_back(old_iter_values[i]);
  }
  Expr new_loop;
  VLOG(3) << "changed_loop_num is : " << changed_loop_num;
  VLOG(3) << "old_iter_values.size() is : " << old_iter_values.size();
  if (changed_loop_num >= static_cast<int>(old_iter_values.size())) {
    new_loop = ir::ir_utils::IRCopy(block);
    new_loop.As<ir::ScheduleBlockRealize>()->iter_values = new_iter_values;
  } else {
    if (!old_iter_values[changed_loop_num].as_var()) {
      os << "iter_vars[" << changed_loop_num
         << "] in Expr param(block) must be vars!\n";
      throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
    }

    auto old_var = old_iter_values[changed_loop_num].as_var_ref();
    auto find_partial_loop = ir::ir_utils::CollectIRNodesWithoutTensor(
        expr,
        [&](const Expr* x) {
          return x->As<ir::For>() &&
                 x->As<ir::For>()->loop_var->name == old_var->name &&
                 Contains(*x, block);
        },
        true);
    if (find_partial_loop.size() != 1U) {
      os << "Number of loop with iter_var which is " << old_var->name
         << " should be 1 in Exprs[0] of module_expr!\n";
      throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
    }
    new_loop = ir::ir_utils::IRCopy(*find_partial_loop.begin());
    auto find_schedule_block = ir::ir_utils::CollectIRNodesWithoutTensor(
        new_loop,
        [&](const Expr* x) { return x->As<ir::ScheduleBlockRealize>(); },
        true);
    if (find_schedule_block.size() != 1U) {
      os << "Number of ScheduleBlockRealize in partial_loop should be 1!\n";
      throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
    }

    Expr sch_block = (*find_schedule_block.begin());
    sch_block.As<ir::ScheduleBlockRealize>()->iter_values = new_iter_values;
  }
  VLOG(3) << "new_loop is : " << new_loop;
  if (used_target_loops.empty()) {
    os << "Cannot CopyTransformAndLoopInfo since there is no loop which use "
          "vars in the new_iter_values in Expr[0] of module_expr!";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  Expr res;
  if (used_target_loops.size() == 1) {
    auto for_loop = used_target_loops[0].As<ir::For>();
    res = For::Make(for_loop->loop_var,
                    for_loop->min,
                    for_loop->extent,
                    for_loop->for_type(),
                    for_loop->device_api,
                    new_loop,
                    for_loop->vectorize_info(),
                    for_loop->bind_info());
  } else {
    Expr outer_loop = used_target_loops.front();
    Expr inner_loop = used_target_loops.back();
    inner_loop.As<ir::For>()->body = Block::Make({new_loop});
    res = outer_loop;
  }
  VLOG(3) << "res is : " << res;
  std::vector<Expr> all_loops = this->GetLoops(block);
  if (all_loops.empty()) {
    os << "Cannot CopyTransformAndLoopInfo since there is no loop in Expr "
          "param(block)!";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }
  this->Replace(all_loops[0], res);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::CopyTransformAndLoopInfo(
    const std::string& block_name, const std::string& block_target_name) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "CopyTransformAndLoopInfo";
  std::ostringstream os;
  auto block = this->GetBlock(block_name);
  auto block_target = this->GetBlock(block_target_name);
  this->CopyTransformAndLoopInfo(block, block_target);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

Expr DyScheduleImpl::SampleCategorical(
    utils::LinearRandomEngine::StateType* rand_seed,
    const std::vector<int>& candidates,
    const std::vector<float>& probs) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "SampleCategorical";
  std::ostringstream os;
  if (candidates.size() != probs.size()) {
    os << "vector<int> params(candidates) and vector<int> params(probs) must "
          "have same size in SampleCategorical!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  int seed_idx = utils::SampleDiscreteFromDistribution(probs, rand_seed);
  auto result = candidates[seed_idx];
  Expr result_expr(result);
  return result_expr;
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

std::vector<Expr> DyScheduleImpl::SamplePerfectTile(
    utils::LinearRandomEngine::StateType* rand_seed,
    const Expr& loop,
    int n,
    int max_innermost_factor) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "SamplePerfectTile";
  std::ostringstream os;
  if (!loop.As<ir::For>()) {
    os << "Expr param(loop) should be a For loop";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  if (n < 2) {
    os << "The number of tile factors should be at least 2";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  if (max_innermost_factor < 1) {
    os << "The max innermost factor should be at least 1";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  if (!cinn::common::is_zero(loop.As<ir::For>()->min)) {
    os << "The For loop should start from 0";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  int loop_extent = GetLoopExtent(loop);
  std::vector<int> innermost_factors;
  for (int i = max_innermost_factor; i >= 1; --i) {
    if (loop_extent % i == 0) {
      innermost_factors.push_back(i);
    }
  }
  if (innermost_factors.empty()) {
    os << "No innermost factor found";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }
  int innermost_factor = innermost_factors[utils::SampleUniformInt(
      0, innermost_factors.size(), rand_seed)];
  auto result = SampleTile(rand_seed, n - 1, loop_extent / innermost_factor);
  std::vector<Expr> result_expr;
  for (auto& factor : result) {
    result_expr.push_back(Expr(factor));
  }
  result_expr.push_back(Expr(innermost_factor));
  return result_expr;
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

Expr DyScheduleImpl::AddUnitLoop(const Expr& block) const {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "AddUnitLoop";
  std::ostringstream os;
  auto exprs = module_expr_.GetExprs();
  return analyzer::AddUnitLoop(exprs, block);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

}  // namespace ir
}  // namespace cinn
