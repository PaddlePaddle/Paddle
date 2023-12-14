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
#define CINN_IR_SCHEDULE_END(err_msg_level)                    \
  }                                                            \
  catch (const utils::ErrorHandler& err_hanlder) {             \
    CINN_THROW(err_hanlder.FormatErrorMessage(err_msg_level)); \
  }

namespace cinn {
namespace ir {

void DyScheduleImpl::MergeExprs() {
  auto exprs = this->GetModule().GetExprs();
  if (exprs.size() == 1U) return;
  CHECK(exprs[0].As<ir::Block>());
  CHECK_EQ(exprs[0].As<ir::Block>()->stmts.size(), 1U);
  CHECK(exprs[0].As<ir::Block>()->stmts[0].As<ir::ScheduleBlockRealize>());
  CHECK(exprs[0]
            .As<ir::Block>()
            ->stmts[0]
            .As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>());
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
    CHECK_EQ(root_block.size(), 1U);
    for (auto& it_block : root_block) {
      auto& block_body = it_block.As<ir::ScheduleBlockRealize>()
                             ->schedule_block.As<ir::ScheduleBlock>()
                             ->body;
      merged_block.push_back(block_body);
    }
  }
  for (auto& block : merged_block) {
    VLOG(3) << "in merged_block, it has " << block;
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
}

bool DyScheduleImpl::HasBlock(const std::string& block_name) const {
  auto exprs = module_expr_.GetExprs();
  return analyzer::HasBlock(exprs, block_name);
}

std::vector<Expr> DyScheduleImpl::GetLoops(const Expr& block) const {
  auto exprs = module_expr_.GetExprs();
  return analyzer::GetLoops(exprs, block);
}

std::vector<Expr> DyScheduleImpl::GetLoops(
    const std::string& block_name) const {
  auto exprs = module_expr_.GetExprs();
  return analyzer::GetLoops(exprs, block_name);
}

std::vector<Expr> DyScheduleImpl::GetAllBlocks() const {
  auto exprs = module_expr_.GetExprs();
  return analyzer::GetAllBlocks(exprs);
}

std::vector<Expr> DyScheduleImpl::GetChildBlocks(const Expr& expr) const {
  return analyzer::GetChildBlocks(expr);
}

Expr DyScheduleImpl::GetBlock(const std::string& block_name) const {
  auto exprs = module_expr_.GetExprs();
  return analyzer::GetBlock(exprs, block_name);
}

Expr DyScheduleImpl::GetRootBlock(const Expr& expr) const {
  auto exprs = module_expr_.GetExprs();
  return analyzer::GetRootBlock(exprs, expr);
}

DeviceAPI DyScheduleImpl::GetDeviceAPI() const {
  auto exprs = module_expr_.GetExprs();
  return analyzer::GetDeviceAPI(exprs);
}

void DyScheduleImpl::Annotate(const Expr& block,
                              const std::string& key,
                              const attr_t& value) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(block.As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>());
  auto copied_block = ir::ir_utils::IRCopy(block);
  auto* schedule_block = copied_block.As<ir::ScheduleBlockRealize>()
                             ->schedule_block.As<ir::ScheduleBlock>();
  schedule_block->attrs.emplace(key, value);
  this->Replace(block, copied_block);
}

void DyScheduleImpl::Unannotate(Expr& block,
                                const std::string& ann_key) {  // NOLINT
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(block.As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>());
  auto* schedule_block = block.As<ir::ScheduleBlockRealize>()
                             ->schedule_block.As<ir::ScheduleBlock>();
  if (schedule_block->attrs.count(ann_key)) {
    schedule_block->attrs.erase(ann_key);
  } else {
    LOG(WARNING) << "Can't find annotation with key: " << ann_key;
    return;
  }
}

void DyScheduleImpl::CopyTransformAndLoopInfo(const Expr& block,
                                              const Expr& block_target) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(block_target.As<ir::ScheduleBlockRealize>());
  auto exprs = this->GetModule().GetExprs();
  CHECK_EQ(exprs.size(), 1U);
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
    CHECK(vars[i]->upper_bound.defined() &&
          vars_target[i]->upper_bound.defined());
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

  if (new_iter_values.empty())
    LOG(FATAL) << "Cannot CopyTransformAndLoopInfo since shape[0] of source "
                  "and target is not equal! "
               << vars[0]->upper_bound << " v.s "
               << vars_target[0]->upper_bound;

  int changed_loop_num = new_iter_values.size();
  std::set<std::string> used_target_loop_vars;
  for (auto& iter_val : new_iter_values) {
    auto find_partial_loop =
        ir::ir_utils::CollectIRNodesWithoutTensor(iter_val, [&](const Expr* x) {
          if (x->as_var()) used_target_loop_vars.insert(x->as_var_ref()->name);
          return x->as_var();
        });
  }
  CHECK(!used_target_loop_vars.empty());
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
    CHECK_EQ(find_loop_var.size(), 1U);
    used_target_loops.push_back(*find_loop_var.begin());
    VLOG(3) << "used_target_loops push_back " << used_target_loops.back();
  }
  std::sort(
      used_target_loops.begin(), used_target_loops.end(), [&](Expr i, Expr j) {
        return (utils::GetStreamCnt(i).size() > utils::GetStreamCnt(j).size());
      });
  for (int i = new_iter_values.size(); i < old_iter_values.size(); ++i) {
    CHECK(old_iter_values[i].as_var());
    new_iter_values.push_back(old_iter_values[i]);
  }
  Expr new_loop;
  VLOG(3) << "changed_loop_num is : " << changed_loop_num;
  VLOG(3) << "old_iter_values.size() is : " << old_iter_values.size();
  if (changed_loop_num >= static_cast<int>(old_iter_values.size())) {
    new_loop = ir::ir_utils::IRCopy(block);
    new_loop.As<ir::ScheduleBlockRealize>()->iter_values = new_iter_values;
  } else {
    CHECK(old_iter_values[changed_loop_num].as_var());
    auto old_var = old_iter_values[changed_loop_num].as_var_ref();
    auto find_partial_loop = ir::ir_utils::CollectIRNodesWithoutTensor(
        expr,
        [&](const Expr* x) {
          return x->As<ir::For>() &&
                 x->As<ir::For>()->loop_var->name == old_var->name &&
                 Contains(*x, block);
        },
        true);
    CHECK_EQ(find_partial_loop.size(), 1U);
    new_loop = ir::ir_utils::IRCopy(*find_partial_loop.begin());
    auto find_schedule_block = ir::ir_utils::CollectIRNodesWithoutTensor(
        new_loop,
        [&](const Expr* x) { return x->As<ir::ScheduleBlockRealize>(); },
        true);
    CHECK_EQ(find_schedule_block.size(), 1U);
    Expr sch_block = (*find_schedule_block.begin());
    sch_block.As<ir::ScheduleBlockRealize>()->iter_values = new_iter_values;
  }
  VLOG(3) << "new_loop is : " << new_loop;
  CHECK(!used_target_loops.empty());
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
  CHECK(!all_loops.empty());
  this->Replace(all_loops[0], res);
}

void DyScheduleImpl::CopyTransformAndLoopInfo(
    const std::string& block_name, const std::string& block_target_name) {
  auto block = this->GetBlock(block_name);
  auto block_target = this->GetBlock(block_target_name);
  this->CopyTransformAndLoopInfo(block, block_target);
}

Expr DyScheduleImpl::SampleCategorical(
    utils::LinearRandomEngine::StateType* rand_seed,
    const std::vector<int>& candidates,
    const std::vector<float>& probs) {
  CINN_NOT_IMPLEMENTED;
}

std::vector<Expr> DyScheduleImpl::SamplePerfectTile(
    utils::LinearRandomEngine::StateType* rand_seed,
    const Expr& loop,
    int n,
    int max_innermost_factor) {
  CINN_NOT_IMPLEMENTED;
}

Expr DyScheduleImpl::AddUnitLoop(const Expr& block) const {
  auto exprs = module_expr_.GetExprs();
  return analyzer::AddUnitLoop(exprs, block);
}

}  // namespace ir
}  // namespace cinn

namespace cinn {
namespace ir {

void StScheduleImpl::MergeExprs() {
  auto exprs = this->GetModule().GetExprs();
  if (exprs.size() == 1U) return;
  CHECK(exprs[0].As<ir::Block>());
  CHECK_EQ(exprs[0].As<ir::Block>()->stmts.size(), 1U);
  CHECK(exprs[0].As<ir::Block>()->stmts[0].As<ir::ScheduleBlockRealize>());
  CHECK(exprs[0]
            .As<ir::Block>()
            ->stmts[0]
            .As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>());
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
    CHECK_EQ(root_block.size(), 1U);
    for (auto& it_block : root_block) {
      auto& block_body = it_block.As<ir::ScheduleBlockRealize>()
                             ->schedule_block.As<ir::ScheduleBlock>()
                             ->body;
      merged_block.push_back(block_body);
    }
  }
  for (auto& block : merged_block) {
    VLOG(3) << "in merged_block, it has " << block;
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
}

bool StScheduleImpl::HasBlock(const std::string& block_name) const {
  auto exprs = module_expr_.GetExprs();
  return analyzer::HasBlock(exprs, block_name);
}

std::vector<Expr> StScheduleImpl::GetLoops(const Expr& block) const {
  auto exprs = module_expr_.GetExprs();
  return analyzer::GetLoops(exprs, block);
}

std::vector<Expr> StScheduleImpl::GetLoops(
    const std::string& block_name) const {
  auto exprs = module_expr_.GetExprs();
  return analyzer::GetLoops(exprs, block_name);
}

std::vector<Expr> StScheduleImpl::GetAllBlocks() const {
  auto exprs = module_expr_.GetExprs();
  return analyzer::GetAllBlocks(exprs);
}

std::vector<Expr> StScheduleImpl::GetChildBlocks(const Expr& expr) const {
  return analyzer::GetChildBlocks(expr);
}

Expr StScheduleImpl::GetBlock(const std::string& block_name) const {
  auto exprs = module_expr_.GetExprs();
  return analyzer::GetBlock(exprs, block_name);
}

Expr StScheduleImpl::GetRootBlock(const Expr& expr) const {
  auto exprs = module_expr_.GetExprs();
  return analyzer::GetRootBlock(exprs, expr);
}

DeviceAPI StScheduleImpl::GetDeviceAPI() const {
  auto exprs = module_expr_.GetExprs();
  return analyzer::GetDeviceAPI(exprs);
}

void StScheduleImpl::Annotate(const Expr& block,
                              const std::string& key,
                              const attr_t& value) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(block.As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>());
  auto copied_block = ir::ir_utils::IRCopy(block);
  auto* schedule_block = copied_block.As<ir::ScheduleBlockRealize>()
                             ->schedule_block.As<ir::ScheduleBlock>();
  schedule_block->attrs.emplace(key, value);
  this->Replace(block, copied_block);
}

void StScheduleImpl::Unannotate(Expr& block, const std::string& ann_key) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(block.As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>());
  auto* schedule_block = block.As<ir::ScheduleBlockRealize>()
                             ->schedule_block.As<ir::ScheduleBlock>();
  if (schedule_block->attrs.count(ann_key)) {
    schedule_block->attrs.erase(ann_key);
  } else {
    LOG(WARNING) << "Can't find annotation with key: " << ann_key;
    return;
  }
}

void StScheduleImpl::CopyTransformAndLoopInfo(
    const std::string& block_name, const std::string& block_target_name) {
  auto block = this->GetBlock(block_name);
  auto block_target = this->GetBlock(block_target_name);
  this->CopyTransformAndLoopInfo(block, block_target);
}

void StScheduleImpl::CopyTransformAndLoopInfo(const Expr& block,
                                              const Expr& block_target) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(block_target.As<ir::ScheduleBlockRealize>());
  auto exprs = this->GetModule().GetExprs();
  CHECK_EQ(exprs.size(), 1U);
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
    CHECK(vars[i]->upper_bound.defined() &&
          vars_target[i]->upper_bound.defined());
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

  if (new_iter_values.empty())
    LOG(FATAL) << "Cannot CopyTransformAndLoopInfo since shape[0] of source "
                  "and target is not equal! "
               << vars[0]->upper_bound << " v.s "
               << vars_target[0]->upper_bound;

  int changed_loop_num = new_iter_values.size();
  std::set<std::string> used_target_loop_vars;
  for (auto& iter_val : new_iter_values) {
    auto find_partial_loop =
        ir::ir_utils::CollectIRNodesWithoutTensor(iter_val, [&](const Expr* x) {
          if (x->as_var()) used_target_loop_vars.insert(x->as_var_ref()->name);
          return x->as_var();
        });
  }
  CHECK(!used_target_loop_vars.empty());
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
    CHECK_EQ(find_loop_var.size(), 1U);
    used_target_loops.push_back(*find_loop_var.begin());
    VLOG(3) << "used_target_loops push_back " << used_target_loops.back();
  }
  std::sort(
      used_target_loops.begin(), used_target_loops.end(), [&](Expr i, Expr j) {
        return (utils::GetStreamCnt(i).size() > utils::GetStreamCnt(j).size());
      });
  for (int i = new_iter_values.size(); i < old_iter_values.size(); ++i) {
    CHECK(old_iter_values[i].as_var());
    new_iter_values.push_back(old_iter_values[i]);
  }
  Expr new_loop;
  VLOG(3) << "changed_loop_num is : " << changed_loop_num;
  VLOG(3) << "old_iter_values.size() is : " << old_iter_values.size();
  if (changed_loop_num >= static_cast<int>(old_iter_values.size())) {
    new_loop = ir::ir_utils::IRCopy(block);
    new_loop.As<ir::ScheduleBlockRealize>()->iter_values = new_iter_values;
  } else {
    CHECK(old_iter_values[changed_loop_num].as_var());
    auto old_var = old_iter_values[changed_loop_num].as_var_ref();
    auto find_partial_loop = ir::ir_utils::CollectIRNodesWithoutTensor(
        expr,
        [&](const Expr* x) {
          return x->As<ir::For>() &&
                 x->As<ir::For>()->loop_var->name == old_var->name &&
                 Contains(*x, block);
        },
        true);
    CHECK_EQ(find_partial_loop.size(), 1U);
    new_loop = ir::ir_utils::IRCopy(*find_partial_loop.begin());
    auto find_schedule_block = ir::ir_utils::CollectIRNodesWithoutTensor(
        new_loop,
        [&](const Expr* x) { return x->As<ir::ScheduleBlockRealize>(); },
        true);
    CHECK_EQ(find_schedule_block.size(), 1U);
    Expr sch_block = (*find_schedule_block.begin());
    sch_block.As<ir::ScheduleBlockRealize>()->iter_values = new_iter_values;
  }
  VLOG(3) << "new_loop is : " << new_loop;
  CHECK(!used_target_loops.empty());
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
  CHECK(!all_loops.empty());
  this->Replace(all_loops[0], res);
}

std::vector<Expr> StScheduleImpl::SamplePerfectTile(
    utils::LinearRandomEngine::StateType* rand_seed,
    const Expr& loop,
    int n,
    int max_innermost_factor) {
  CHECK(loop.As<ir::For>())
      << "Expr param of SamplePerfectTile should be a For loop";
  CHECK_GE(n, 2) << "The number of tile factors should be at least 2";
  CHECK_GE(max_innermost_factor, 1)
      << "The max innermost factor should be at least 1";
  CHECK(cinn::common::is_zero(loop.As<ir::For>()->min))
      << "The For loop should start from 0";
  int loop_extent = GetLoopExtent(loop);
  std::vector<int> innermost_factors;
  for (int i = max_innermost_factor; i >= 1; --i) {
    if (loop_extent % i == 0) {
      innermost_factors.push_back(i);
    }
  }
  CHECK(!innermost_factors.empty()) << "No innermost factor found";
  int innermost_factor = innermost_factors[utils::SampleUniformInt(
      0, innermost_factors.size(), rand_seed)];
  auto result = SampleTile(rand_seed, n - 1, loop_extent / innermost_factor);
  std::vector<Expr> result_expr;
  for (auto& factor : result) {
    result_expr.push_back(Expr(factor));
  }
  result_expr.push_back(Expr(innermost_factor));
  return result_expr;
}

Expr StScheduleImpl::SampleCategorical(
    utils::LinearRandomEngine::StateType* rand_seed,
    const std::vector<int>& candidates,
    const std::vector<float>& probs) {
  // check two sizes
  CHECK_EQ(candidates.size(), probs.size())
      << "candidates and probs must have same size.";
  int seed_idx = utils::SampleDiscreteFromDistribution(probs, rand_seed);
  auto result = candidates[seed_idx];
  Expr result_expr(result);
  return result_expr;
}

Expr StScheduleImpl::AddUnitLoop(const Expr& block) const {
  auto exprs = module_expr_.GetExprs();
  return analyzer::AddUnitLoop(exprs, block);
}

}  // namespace ir
}  // namespace cinn
