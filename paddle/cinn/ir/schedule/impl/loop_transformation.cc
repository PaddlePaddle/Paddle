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

#include "paddle/cinn/ir/schedule/impl/ir_schedule.h"

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/common/macros.h"
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
#define CINN_IR_SCHEDULE_END(err_msg_level)                                 \
  }                                                                         \
  catch (const utils::ErrorHandler& err_handler) {                          \
    PADDLE_THROW(                                                           \
        phi::errors::Fatal(err_handler.FormatErrorMessage(err_msg_level))); \
  }

namespace cinn {
namespace ir {

std::vector<Expr> DyScheduleImpl::Split(const Expr& loop,
                                        const std::vector<int>& factors) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "Split";
  std::ostringstream os;

  if (!loop.As<ir::For>()) {
    os << "Expr param(loop) must be For node! Please check!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }
  auto* for_node = loop.As<ir::For>();
  if (!cinn::common::is_zero(for_node->min)) {
    os << "The For node must start with 0! Please check!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }
  if (factors.empty()) {
    os << "The factors param of Split should not be empty! Please check!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  if (loop.As<For>()->extent.is_constant()) {
    int tot_extent = for_node->extent.get_constant();

    VLOG(3) << "Try Split loop from (" << for_node->loop_var->name << ", 0, "
            << tot_extent << ") to (" << cinn::utils::Join(factors, ", ")
            << ") at loop:\n"
            << loop;

    std::vector<int> processed_factors;
    processed_factors =
        ValidateFactors(factors, tot_extent, this->module_expr_);
    int prod_size = std::accumulate(processed_factors.begin(),
                                    processed_factors.end(),
                                    1,
                                    std::multiplies<int>());
    std::vector<Var> new_loop_vars;
    Expr substitute_value(0);
    for (int i = 0; i < processed_factors.size(); ++i) {
      Var temp_var(cinn::common::UniqName(for_node->loop_var->name));
      substitute_value =
          Expr(temp_var) + substitute_value * Expr(processed_factors[i]);
      new_loop_vars.push_back(temp_var);
    }
    substitute_value = cinn::common::AutoSimplify(substitute_value);
    Expr new_node = ir::ir_utils::IRCopy(for_node->body);
    ReplaceExpr(&new_node, {for_node->loop_var}, {substitute_value});
    std::vector<Expr> splited_loops;
    splited_loops.resize(processed_factors.size());
    if (tot_extent < prod_size) {
      new_node = IfThenElse::Make(LT::Make(substitute_value, for_node->extent),
                                  new_node);
    }
    for (int i = processed_factors.size() - 1; i >= 0; i--) {
      if (!new_node.As<ir::Block>()) new_node = Block::Make({new_node});
      new_node = For::Make(new_loop_vars[i],
                           Expr(0),
                           Expr(processed_factors[i]),
                           for_node->for_type(),
                           for_node->device_api,
                           new_node);
      splited_loops[i] = new_node;
    }

    this->Replace(loop, new_node);
    VLOG(3) << "After Split, ir is:\n" << splited_loops.at(0);
    return splited_loops;
  }

  Expr tot_extent = for_node->extent;

  VLOG(3) << "Try Split loop from (" << for_node->loop_var->name << ", 0, "
          << tot_extent << ") to (" << cinn::utils::Join(factors, ", ")
          << ") at loop:\n"
          << loop;

  bool is_positive = true;
  int num_minus1 = 0;
  std::vector<Expr> process_factors;
  Expr prod_size(-1);
  int idx_neg1 = 1;
  for (auto factor : factors) prod_size = prod_size * Expr(factor);
  std::for_each(factors.begin(), factors.end(), [&](int factor) {
    if (factor == -1) {
      process_factors.push_back(
          cinn::common::AutoSimplify(tot_extent / prod_size));
      idx_neg1 = -idx_neg1;
    } else {
      process_factors.push_back(Expr(factor));
      if (idx_neg1 > 0) idx_neg1++;
    }
    if (factor < 1 && factor != -1) is_positive = false;
    if (factor == -1) ++num_minus1;
  });

  idx_neg1 = (-idx_neg1) - 1;

  bool exact_split =
      (tot_extent ==
       cinn::common::AutoSimplify(process_factors[0] * process_factors[1]));
  if (!exact_split) {
    process_factors[idx_neg1] =
        cinn::common::AutoSimplify(process_factors[idx_neg1] + Expr(1));
  }

  if (num_minus1 > 1 || (!is_positive)) {
    os << "The params in factors of Split on dynamic shape should contains at "
          "most one '-1' and the rest of them should be positive!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  std::vector<Var> new_loop_vars;
  Expr substitute_value(0);
  for (int i = 0; i < process_factors.size(); ++i) {
    Var temp_var(common::UniqName(for_node->loop_var->name));
    substitute_value = Expr(temp_var) + substitute_value * process_factors[i];
    new_loop_vars.push_back(temp_var);
  }
  substitute_value = cinn::common::AutoSimplify(substitute_value);
  Expr new_node = ir::ir_utils::IRCopy(for_node->body);
  ReplaceExpr(&new_node, {for_node->loop_var}, {substitute_value});
  std::vector<Expr> splited_loops;
  splited_loops.resize(process_factors.size());

  if (!exact_split) {
    new_node =
        IfThenElse::Make(LT::Make(substitute_value, tot_extent), new_node);
  }

  for (int i = process_factors.size() - 1; i >= 0; i--) {
    if (!new_node.As<ir::Block>()) new_node = Block::Make({new_node});
    new_node = For::Make(new_loop_vars[i],
                         Expr(0),
                         process_factors[i],
                         for_node->for_type(),
                         for_node->device_api,
                         new_node);
    splited_loops[i] = new_node;
  }

  this->Replace(loop, new_node);
  VLOG(3) << "After Split, ir is:\n" << splited_loops.at(0);
  return splited_loops;
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

// TODO(@LiuYang): now -1 can't exist in factors.
std::vector<Expr> DyScheduleImpl::Split(const Expr& loop,
                                        const std::vector<Expr>& factors) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "Split";
  std::ostringstream os;
  if (!loop.As<ir::For>()) {
    os << "Expr param(loop) must be For node! Please check!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  auto* for_node = loop.As<ir::For>();

  if (!common::is_zero(for_node->min)) {
    os << "The For node must start with 0! Please check!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }
  if (factors.empty()) {
    os << "The factors param of Split should not be empty! Please check!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  if (loop.As<ir::For>()->extent.is_constant()) {
    os << "Can't Split a loop with constant extent but with variable in "
          "factors! Please check!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  Expr tot_extent = for_node->extent;

  VLOG(3) << "Try Split loop from (" << for_node->loop_var->name << ", 0, "
          << tot_extent << ") to (" << cinn::utils::Join(factors, ", ")
          << ") at loop:\n"
          << loop;

  std::vector<Expr> process_factors(factors);
  Expr prod_size(1);
  for (auto factor : factors) prod_size = prod_size * Expr(factor);
  common::cas_intervals_t var_intervals = {};
  cinn::common::SymbolicExprAnalyzer analyzer(var_intervals);
  if (!analyzer.ProveEQ(tot_extent, prod_size).value_or(false)) {
    os << "Product of factors can't be proved to be equal to the extent of "
          "current for loop! Please check!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  std::vector<Var> new_loop_vars;
  Expr substitute_value(0);
  for (int i = 0; i < process_factors.size(); ++i) {
    Var temp_var(common::UniqName(for_node->loop_var->name));
    substitute_value = Expr(temp_var) + substitute_value * process_factors[i];
    new_loop_vars.push_back(temp_var);
  }
  substitute_value = cinn::common::AutoSimplify(substitute_value);
  Expr new_node = ir::ir_utils::IRCopy(for_node->body);
  ReplaceExpr(&new_node, {for_node->loop_var}, {substitute_value});
  std::vector<Expr> splited_loops;
  splited_loops.resize(process_factors.size());

  for (int i = process_factors.size() - 1; i >= 0; i--) {
    if (!new_node.As<ir::Block>()) new_node = Block::Make({new_node});
    new_node = For::Make(new_loop_vars[i],
                         Expr(0),
                         process_factors[i],
                         for_node->for_type(),
                         for_node->device_api,
                         new_node);
    splited_loops[i] = new_node;
  }

  this->Replace(loop, new_node);
  VLOG(3) << "After Split, ir is:\n" << splited_loops.at(0);
  return splited_loops;
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

Expr DyScheduleImpl::Fuse(const std::vector<Expr>& loops) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "Fuse";
  std::ostringstream os;

  VLOG(3) << "Tring to fuse:\n" << cinn::utils::Join(loops, "\n");
  std::vector<const ir::For*> for_nodes;
  std::vector<Var> loop_vars;
  if (loops.empty()) {
    os << "The loops param of Fuse should not be empty! Please check!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  for (const Expr& it_loop : loops) {
    if (!it_loop.As<ir::For>()) {
      os << "Loop in vector<Expr> param(loops) of Fuse must be For node! "
            "Please check!\n";
      throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
    }

    if (!for_nodes.empty()) {
      if (!for_nodes.back()->body.As<ir::Block>()) {
        os << "The body of for node is not Block! Please check!\n";
        throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
      }

      if (for_nodes.back()->body.As<ir::Block>()->stmts.size() != 1) {
        os << "The Block's size of for node is not 1! Please check!\n";
        throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
      }

      if (for_nodes.back()->body.As<ir::Block>()->stmts[0] != it_loop) {
        os << "The For nodes in loops param of Fuse must be adjacent! Please "
              "check!\n";
        throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
      }
    }
    for_nodes.push_back(it_loop.As<ir::For>());
    loop_vars.push_back(it_loop.As<ir::For>()->loop_var);
  }
  std::string suffix;
  suffix = for_nodes[0]->loop_var->name;
  int loops_number = for_nodes.size();
  for (int i = 1; i < loops_number; ++i) {
    suffix += "_" + for_nodes[i]->loop_var->name;
  }
  suffix += "_fused";
  Var fused_var(suffix);
  std::vector<Expr> substitute_value;
  substitute_value.resize(loops_number);
  Expr fused_expr(fused_var);
  for (int i = loops_number - 1; i > 0; i--) {
    substitute_value[i] = Mod::Make(fused_expr, for_nodes[i]->extent);
    fused_expr = Div::Make(fused_expr, for_nodes[i]->extent);
  }
  substitute_value[0] = fused_expr;

  Expr fused_body = ir::ir_utils::IRCopy(for_nodes.back()->body);
  ReplaceExpr(&fused_body, loop_vars, substitute_value);
  optim::Simplify(&fused_body);
  Expr fused_extent(1);
  for (int i = 0; i < loops_number; ++i) {
    fused_extent = fused_extent * for_nodes[i]->extent;
  }
  fused_extent = cinn::common::AutoSimplify(fused_extent);
  if (!fused_body.As<ir::Block>()) fused_body = Block::Make({fused_body});
  Expr new_stmt = For::Make(fused_var,
                            Expr(0),
                            fused_extent,
                            for_nodes[0]->for_type(),
                            for_nodes[0]->device_api,
                            fused_body);
  this->Replace(loops[0], new_stmt);

  VLOG(3) << "After fuse, ir is:\n" << new_stmt;
  return new_stmt;
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

Expr DyScheduleImpl::Fuse(const std::string& block_name,
                          const std::vector<int>& loops_index) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "Fuse";
  std::ostringstream os;
  std::vector<Expr> all_loops = this->GetLoops(block_name);
  std::vector<Expr> loops_expr;
  loops_expr.reserve(loops_index.size());
  for (int i = 0; i < loops_index.size(); ++i) {
    if (i > 0) {
      if (loops_index[i - 1] + 1 != loops_index[i]) {
        os << "Loops index in Fuse should be continuous!\n";
        throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
      }
    }
  }
  for (int i : loops_index) {
    if (i >= static_cast<int>(all_loops.size())) {
      os << "The loop index in Fuse should be less than total loop's number!\n";
      throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
    }

    if (i < 0) {
      os << "The loop index in Fuse should be >= 0!\n";
      throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
    }
    loops_expr.emplace_back(all_loops[i]);
  }
  return this->Fuse(loops_expr);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

Expr DyScheduleImpl::Fuse(const Expr& block,
                          const std::vector<int>& loops_index) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "Fuse";
  std::ostringstream os;
  std::vector<Expr> all_loops = this->GetLoops(block);
  std::vector<Expr> loops_expr;
  loops_expr.reserve(loops_index.size());
  for (int i = 0; i < loops_index.size(); ++i) {
    if (i > 0) {
      if (loops_index[i - 1] + 1 != loops_index[i]) {
        os << "Loops index in Fuse should be continuous!\n";
        throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
      }
    }
  }
  for (int i : loops_index) {
    if (i >= static_cast<int>(all_loops.size())) {
      os << "The loop index in Fuse should be less than total loop's number!\n";
      throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
    }

    if (i <= 0) {
      os << "The loop index in Fuse should be >= 0!\n";
      throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
    }
    loops_expr.emplace_back(all_loops[i]);
  }
  return this->Fuse(loops_expr);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

Expr DyScheduleImpl::Reorder(const std::vector<Expr>& loops) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "Reorder";
  std::ostringstream os;
  if (loops.size() <= 1) {
    return Expr{nullptr};
  }
  VLOG(4) << "Before Reorder, ir is:\n" << loops[0];

  std::set<Expr, CompExpr> loop_set = CollectLoopsToSet(loops);
  auto boundary = GetBoundaryOfReorderRange(loop_set);
  Expr top = boundary.first;
  Expr bottom = boundary.second;
  std::vector<Expr> chain = GetLoopsInRange(top, bottom);
  std::vector<Expr> if_nodes = GetIfThenElseInRange(top, bottom);
  Expr new_loop = ConstructNewLoopChain(chain, loops, loop_set, if_nodes);
  this->Replace(top, new_loop);

  VLOG(4) << "After Reorder, ir is:\n" << new_loop;
  return new_loop;
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

Expr DyScheduleImpl::Reorder(const std::string& block_name,
                             const std::vector<int>& loops_index) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "Reorder";
  std::ostringstream os;

  std::vector<Expr> all_loops = this->GetLoops(block_name);
  std::vector<Expr> loops_expr;
  loops_expr.reserve(loops_index.size());
  for (int i : loops_index) {
    if (i >= static_cast<int>(all_loops.size())) {
      os << "The loop index in Reorder should be less than total loop's "
            "number!\n";
      throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
    }

    if (i < 0) {
      os << "The loop index in Reorder should be >= 0!\n";
      throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
    }
    loops_expr.emplace_back(all_loops[i]);
  }
  return this->Reorder(loops_expr);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

Expr DyScheduleImpl::Reorder(const Expr& block,
                             const std::vector<int>& loops_index) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "Reorder";
  std::ostringstream os;

  std::vector<Expr> all_loops = this->GetLoops(block);
  std::vector<Expr> loops_expr;
  loops_expr.reserve(loops_index.size());
  for (int i : loops_index) {
    if (i >= static_cast<int>(all_loops.size())) {
      os << "The loop index in Reorder should be less than total loop's "
            "number!\n";
      throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
    }

    if (i < 0) {
      os << "The loop index in Reorder should be >= 0!\n";
      throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
    }

    loops_expr.emplace_back(all_loops[i]);
  }
  return this->Reorder(loops_expr);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::FlattenLoops(const std::vector<Expr>& loops,
                                  const bool force_flat) {
  CINN_NOT_IMPLEMENTED;
}

}  // namespace ir
}  // namespace cinn
