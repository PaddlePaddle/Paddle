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
  for (auto factor : factors) prod_size = prod_size * Expr(factor);
  std::for_each(factors.begin(), factors.end(), [&](int factor) {
    if (factor == -1) {
      process_factors.push_back(
          cinn::common::AutoSimplify(tot_extent / prod_size + Expr(1)));
    } else {
      process_factors.push_back(Expr(factor));
    }
    if (factor < 1 && factor != -1) is_positive = false;
    if (factor == -1) ++num_minus1;
  });

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

  new_node = IfThenElse::Make(LT::Make(substitute_value, tot_extent), new_node);

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

// TODO(@LiuYang): now -1 can't exsit in factors,
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

void DyScheduleImpl::Broadcast(const std::string& block_name,
                               const BroadcastInfo& info) {
  auto axes = info.broadcast_axes;
  std::vector<Expr> all_loops = this->GetLoops(block_name);
  if (axes[0] >= all_loops.size()) {
    throw std::runtime_error("axes execeed loop size");
  }
  auto broadcast_loop = all_loops[axes[0]].As<ir::For>();

  Expr broadcast_body = ir::ir_utils::IRCopy(broadcast_loop->body);

  auto schedule_realize = broadcast_body.As<ir::Block>()
                              ->expr_fields()[0]
                              ->As<ir::ScheduleBlockRealize>();

  auto schedule_block =
      schedule_realize->schedule_block.As<ir::ScheduleBlock>();

  auto iter_vars = schedule_block->iter_vars;
  auto iter_values = schedule_realize->iter_values;
  schedule_realize->iter_values[axes[0]] = broadcast_loop->loop_var;

  auto exprs = ir::ir_utils::CollectIRNodesInOrder(
      schedule_block->body, [&](const Expr* x) { return x->As<ir::Load>(); });

  for (auto expr : exprs) {
    ReplaceExpr(&expr, {schedule_block->iter_vars[axes[0]]}, {Expr(0)});
  }

  auto factors = info.output_shape;
  int factor = factors[0];
  Expr new_extent(factor);

  if (!broadcast_body.As<ir::Block>())
    broadcast_body = Block::Make({broadcast_body});
  Expr new_stmt = For::Make(broadcast_loop->loop_var,
                            Expr(0),
                            new_extent,
                            broadcast_loop->for_type(),
                            broadcast_loop->device_api,
                            broadcast_body);

  this->Replace(broadcast_loop, new_stmt);
}

void DyScheduleImpl::BroadcastToElementwise(const std::string& block_name,
                                            const std::vector<int64_t>& axes) {
  std::vector<Expr> all_loops = this->GetLoops(block_name);

  auto broadcast_loop = all_loops[axes[0]].As<ir::For>();

  Expr broadcast_body = broadcast_loop->body;

  auto schedule_realize = broadcast_body.As<ir::Block>()
                              ->expr_fields()[0]
                              ->As<ir::ScheduleBlockRealize>();

  auto schedule_block =
      schedule_realize->schedule_block.As<ir::ScheduleBlock>();

  auto iter_vars = schedule_block->iter_vars;

  auto iter_values = schedule_realize->iter_values;
  auto exprs = ir::ir_utils::CollectIRNodesInOrder(
      schedule_block->body, [&](const Expr* x) { return x->As<ir::Load>(); });

  for (auto expr : exprs) {
    auto load = expr.As<ir::Load>();
    load->indices[axes[0]] = broadcast_loop->loop_var;
  }
}

}  // namespace ir
}  // namespace cinn

namespace cinn {
namespace ir {

std::vector<Expr> StScheduleImpl::Split(const Expr& loop,
                                        const std::vector<int>& factors) {
  CHECK(loop.As<ir::For>())
      << "Expr param of Split must be For node! Please check.";
  auto* for_node = loop.As<ir::For>();
  CHECK(cinn::common::is_zero(for_node->min))
      << "The For node must start with 0! Please check.";
  CHECK(for_node->extent.is_constant())
      << "The For node's extent must be constant! Please check.";
  int tot_extent = for_node->extent.get_constant();

  VLOG(3) << "Try Split loop from (" << for_node->loop_var->name << ", 0, "
          << tot_extent << ") to (" << cinn::utils::Join(factors, ", ")
          << ") at loop:\n"
          << loop;

  std::vector<int> processed_factors;
  CINN_IR_SCHEDULE_BEGIN();
  processed_factors = ValidateFactors(factors, tot_extent, this->module_expr_);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
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

void StScheduleImpl::BroadcastToElementwise(const std::string& block_name,
                                            const std::vector<int64_t>& axes) {
  std::vector<Expr> all_loops = this->GetLoops(block_name);

  auto broadcast_loop = all_loops.back().As<ir::For>();

  Expr broadcast_body = broadcast_loop->body;

  auto schedule_realize = broadcast_body.As<ir::Block>()
                              ->expr_fields()[0]
                              ->As<ir::ScheduleBlockRealize>();

  auto schedule_block =
      schedule_realize->schedule_block.As<ir::ScheduleBlock>();

  auto iter_vars = schedule_block->iter_vars;

  auto iter_values = schedule_realize->iter_values;

  auto exprs = ir::ir_utils::CollectIRNodesInOrder(
      schedule_block->body, [&](const Expr* x) { return x->As<ir::Load>(); });

  for (auto expr : exprs) {
    auto load = expr.As<ir::Load>();
    load->indices.resize(all_loops.size(), Expr(0));

    for (size_t i = 0; i < axes.size(); ++i) {
      auto loop_temp = all_loops[axes[i]].As<ir::For>();

      load->indices[axes[i]] = schedule_block->iter_vars[axes[i]];
    }
  }
}

void StScheduleImpl::Broadcast(const std::string& block_name,
                               const BroadcastInfo& info) {
  auto axes = info.broadcast_axes;
  std::vector<Expr> all_loops = this->GetLoops(block_name);
  if (axes[0] >= all_loops.size()) {
    throw std::runtime_error("axes execeed loop size");
  }

  // Get Last loop
  auto broadcast_loop = all_loops.back().As<ir::For>();

  Expr broadcast_body = broadcast_loop->body;

  auto schedule_realize = broadcast_body.As<ir::Block>()
                              ->expr_fields()[0]
                              ->As<ir::ScheduleBlockRealize>();

  auto schedule_block =
      schedule_realize->schedule_block.As<ir::ScheduleBlock>();

  auto iter_vars = schedule_block->iter_vars;

  auto iter_values = schedule_realize->iter_values;

  auto factors = info.output_shape;
  auto full_broadcast = info.full_broadcast;
  auto first_broadcast = info.first_broadcast;
  for (size_t i = 0; i < axes.size(); ++i) {
    // new_extent
    auto axis = axes[i];
    auto loop_temp = all_loops[axis].As<ir::For>();
    int extent = factors[i];
    loop_temp->extent = Expr(extent);

    if (!full_broadcast) {
      schedule_realize->iter_values[axis] = loop_temp->loop_var;
    }

    if (info.with_constrain) {
      auto check = ir::EQ::Make(loop_temp->loop_var, Expr(0));
      schedule_block->body = ir::IfThenElse::Make(check, schedule_block->body);
    }
  }

  if (first_broadcast && !full_broadcast) {
    auto exprs = ir::ir_utils::CollectIRNodesInOrder(
        schedule_block->body, [&](const Expr* x) { return x->As<ir::Load>(); });

    if (info.op_name == "cinn_op.reshape") {
      for (auto expr : exprs) {
        auto load = expr.As<ir::Load>();
        for (size_t k = 0; k < load->indices.size(); ++k) {
          for (size_t i = 0; i < axes.size(); ++i) {
            ReplaceExpr(&load->indices[k],
                        {schedule_block->iter_vars[axes[i]]},
                        {Expr(0)});
          }
        }
      }

      return;
    }
    for (auto expr : exprs) {
      auto load = expr.As<ir::Load>();
      if (load->indices.size() == schedule_realize->iter_values.size()) {
        for (size_t i = 0; i < axes.size(); ++i) {
          load->indices[axes[i]] = Expr(0);
        }
      } else if (load->indices.size() < schedule_realize->iter_values.size()) {
        // only one element
        // replace t zeros

        for (size_t k = 0; k < load->indices.size(); ++k) {
          for (size_t i = 0; i < axes.size(); ++i) {
            ReplaceExpr(&load->indices[k],
                        {schedule_block->iter_vars[axes[i]]},
                        {Expr(0)});
          }
        }
      } else {
        throw std::runtime_error("not support broadcast type yet");
      }
    }
  }
}

std::vector<Expr> StScheduleImpl::Split(const Expr& loop,
                                        const std::vector<Expr>& factors) {
  CHECK(false) << "Static shape schedule don't support Split with some "
                  "variables in factors";
}

Expr StScheduleImpl::Fuse(const std::vector<Expr>& loops) {
  VLOG(3) << "Tring to fuse:\n" << cinn::utils::Join(loops, "\n");
  std::vector<const ir::For*> for_nodes;
  std::vector<Var> loop_vars;
  CHECK(!loops.empty())
      << "The loops param of Fuse should not be empty! Please check.";

  for (const Expr& it_loop : loops) {
    CHECK(it_loop.As<ir::For>())
        << "Expr param of Fuse must be For node! Please check.";
    if (!for_nodes.empty()) {
      CHECK(for_nodes.back()->body.As<ir::Block>())
          << "The body of for node is not Block!";
      // CHECK_EQ(for_nodes.back()->body.As<ir::Block>()->stmts.size(), 1U)
      //     << "The Block'size of for node is not 1!";
      // CHECK_EQ(for_nodes.back()->body.As<ir::Block>()->stmts[0], it_loop)
      //     << "The For nodes in loops param of Fuse must be adjacent! Please "
      //        "check.";
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
}

Expr StScheduleImpl::Fuse(const std::string& block_name,
                          const std::vector<int>& loops_index) {
  std::vector<Expr> all_loops = this->GetLoops(block_name);
  std::vector<Expr> loops_expr;
  loops_expr.reserve(loops_index.size());
  for (int i = 0; i < loops_index.size(); ++i) {
    if (i > 0)
      CHECK_EQ(loops_index[i - 1] + 1, loops_index[i])
          << "Loops index in Fuse should be continuous!";
  }
  for (int i : loops_index) {
    CHECK_LT(i, (int)all_loops.size())
        << "The loop index in Fuse should be less than total loop's number.";
    CHECK_GE(i, 0) << "The loop index in Fuse should be >= 0.";
    loops_expr.emplace_back(all_loops[i]);
  }
  return this->Fuse(loops_expr);
}

Expr StScheduleImpl::Fuse(const Expr& block,
                          const std::vector<int>& loops_index) {
  std::vector<Expr> all_loops = this->GetLoops(block);
  std::vector<Expr> loops_expr;
  loops_expr.reserve(loops_index.size());
  for (int i = 0; i < loops_index.size(); ++i) {
    if (i > 0)
      CHECK_EQ(loops_index[i - 1] + 1, loops_index[i])
          << "Loops index in Fuse should be continuous!";
  }
  for (int i : loops_index) {
    CHECK_LT(i, (int)all_loops.size())
        << "The loop index in Fuse should be less than total loop's number.";
    CHECK_GE(i, 0) << "The loop index in Fuse should be >= 0.";
    loops_expr.emplace_back(all_loops[i]);
  }
  return this->Fuse(loops_expr);
}

Expr StScheduleImpl::Reorder(const std::vector<Expr>& loops) {
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
}

Expr StScheduleImpl::Reorder(const std::string& block_name,
                             const std::vector<int>& loops_index) {
  std::vector<Expr> all_loops = this->GetLoops(block_name);
  std::vector<Expr> loops_expr;
  loops_expr.reserve(loops_index.size());
  for (int i : loops_index) {
    CHECK_LT(i, (int)all_loops.size())
        << "The loop index in Reorder should be less than total loop's number.";
    CHECK_GE(i, 0) << "The loop index in Reorder should be >= 0.";
    loops_expr.emplace_back(all_loops[i]);
  }
  return this->Reorder(loops_expr);
}

Expr StScheduleImpl::Reorder(const Expr& block,
                             const std::vector<int>& loops_index) {
  std::vector<Expr> all_loops = this->GetLoops(block);
  std::vector<Expr> loops_expr;
  loops_expr.reserve(loops_index.size());
  for (int i : loops_index) {
    CHECK_LT(i, (int)all_loops.size())
        << "The loop index in Reorder should be less than total loop's number.";
    CHECK_GE(i, 0) << "The loop index in Reorder should be >= 0.";
    loops_expr.emplace_back(all_loops[i]);
  }
  return this->Reorder(loops_expr);
}

void StScheduleImpl::FlattenLoops(const std::vector<Expr>& loops,
                                  const bool flat_tensor) {
  CHECK_GT(loops.size(), 0) << "Loops can't be empty!";
  VLOG(4) << "Before FlattenLoops, ir is:\n" << loops[0];
  // compute loop
  int extent = 1;
  std::vector<int> strides;
  std::vector<ir::Var> loop_vars(loops.size());
  for (int idx = loops.size() - 1; idx >= 0; --idx) {
    strides.insert(strides.begin(), extent);
    extent *= loops[idx].As<ir::For>()->extent.as_int32();
    loop_vars[idx] = loops[idx].As<ir::For>()->loop_var;
  }
  CHECK_EQ(loops.size(), strides.size());

  // create new loop.
  auto last = loops.back().As<ir::For>();
  auto var = ir::Var("flat_i");
  auto _var = ir::Var("_flat_i");
  auto loop = ir::For::Make(var,
                            ir::Expr(0),
                            ir::Expr(extent),
                            last->for_type(),
                            last->device_api,
                            last->body);

  // map loop var to old loop var.
  auto _iter = ir::Expr(_var);
  std::unordered_map<std::string, ir::Expr> loops_to_flat_var_map;
  for (int idx = 0; idx < strides.size(); ++idx) {
    if (strides[idx] == 1) {
      // flat_i_to_loop_var.push_back(_iter);
      loops_to_flat_var_map[loops[idx].As<ir::For>()->loop_var->name] = _iter;
    } else {
      // flat_i_to_loop_var.push_back(_iter / Expr(strides[idx]));
      loops_to_flat_var_map[loops[idx].As<ir::For>()->loop_var->name] =
          _iter / Expr(strides[idx]);
      _iter = _iter % Expr(strides[idx]);
    }
  }

  ir::FindBlocksVisitor visitor;
  auto blocks = visitor(&last->body);
  auto can_do_flat = [](const std::vector<Expr>& indexs,
                        const std::vector<Var>& loop_vars) {
    if (indexs.size() != loop_vars.size()) {
      return false;
    }

    for (int idx = 0; idx < indexs.size(); ++idx) {
      if (!indexs[idx].as_var()) {
        return false;
      } else {
        auto var = indexs[idx].as_var_ref();
        if (var->name != loop_vars[idx]->name) {
          return false;
        }
      }
    }
    return true;
  };

  // change blocks iter value/iter var
  for (auto& block : blocks) {
    auto block_realize = block.As<ir::ScheduleBlockRealize>();
    auto schedule_block = block_realize->schedule_block.As<ir::ScheduleBlock>();

    // checkout loops in orders.
    std::vector<std::string> var_names = {};
    CHECK_GE(block_realize->iter_values.size(), loop_vars.size())
        << "the number of iter bind values must be more than loop vars!";
    for (int idx = 0; idx < block_realize->iter_values.size(); ++idx) {
      auto& iter = block_realize->iter_values[idx];
      if (iter.is_var()) {
        CHECK_EQ(iter.as_var_ref()->name, loop_vars[idx]->name)
            << "loops is not the same order with tensor!";
      } else {
        CHECK(iter.As<IntImm>()) << iter.node_type() << " is not IntImm";
        CHECK_EQ(iter.as_int32(), 0);
      }
    }

    auto exprs = ir::ir_utils::CollectIRNodesInOrder(
        schedule_block->body,
        [&](const Expr* x) { return x->As<ir::Store>() || x->As<ir::Load>(); });
    // reverse exprs from last to first.
    std::reverse(std::begin(exprs), std::end(exprs));

    std::vector<ir::Var> var_to_replace;
    std::vector<ir::Expr> flat_i_to_loop_var;
    // if iter var is more than flat i to loop, there exist dim = 1.
    for (int idx = 0; idx < block_realize->iter_values.size(); ++idx) {
      if (block_realize->iter_values[idx].is_var()) {
        var_to_replace.push_back(schedule_block->iter_vars[idx]);
        auto var_name = block_realize->iter_values[idx].as_var_ref()->name;
        CHECK(loops_to_flat_var_map.count(var_name))
            << "Can't find var name : " << var_name;
        flat_i_to_loop_var.push_back(loops_to_flat_var_map[var_name]);
      } else {
        CHECK_EQ(block_realize->iter_values[idx].as_int32(), 0);
        // insert var -> 0, to replace var to 0.
        var_to_replace.push_back(schedule_block->iter_vars[idx]);
        flat_i_to_loop_var.push_back(Expr(0));
      }
    }
    CHECK_EQ(var_to_replace.size(), flat_i_to_loop_var.size());

    for (auto expr : exprs) {
      if (expr.As<ir::Store>()) {
        auto store = expr.As<ir::Store>();
        if (store->is_addr_tensor()) {
          auto t = store->tensor.as_tensor_ref();
          CHECK(!t->reduce_axis.size());
          auto tsize = std::accumulate(t->shape.begin(),
                                       t->shape.end(),
                                       1,
                                       [](const int sum, const Expr& expr) {
                                         return sum * expr.as_int32();
                                       });
          if ((!flat_tensor &&
               !can_do_flat(store->indices, schedule_block->iter_vars)) ||
              extent != tsize) {
            // just replace indexs
            for (auto& indice : store->indices) {
              if (!indice.is_var()) {
                continue;
              }
              ReplaceExpr(&indice, var_to_replace, flat_i_to_loop_var);
            }
            // compute index and flat tensor.
            store->indices = {store->index()};
            continue;
          }
          // update var and shape
          store->indices = {Expr(_var)};
        }
      } else {
        auto load = expr.As<ir::Load>();
        if (load->is_addr_tensor()) {
          auto t = load->tensor.as_tensor_ref();
          CHECK(!t->reduce_axis.size());
          auto tsize = std::accumulate(t->shape.begin(),
                                       t->shape.end(),
                                       1,
                                       [](const int sum, const Expr& expr) {
                                         return sum * expr.as_int32();
                                       });
          if ((!flat_tensor &&
               !can_do_flat(load->indices, schedule_block->iter_vars)) ||
              extent != tsize) {
            // just replace indexs
            for (auto& indice : load->indices) {
              if (!indice.is_var()) {
                continue;
              }
              ReplaceExpr(&indice, var_to_replace, flat_i_to_loop_var);
            }
            // compute index and flat tensor.
            load->indices = {load->index()};
            continue;
          }
          // update var and shape
          load->indices = {Expr(_var)};
        }
      }
    }
    ReplaceExpr(&schedule_block->body, var_to_replace, flat_i_to_loop_var);

    // update iter values
    auto iter = ir::Expr(var);
    block_realize->iter_values = {iter};

    // update iter_vars
    schedule_block->iter_vars = {_var};
    CHECK_EQ(block_realize->iter_values.size(),
             schedule_block->iter_vars.size());
  }

  this->Replace(loops[0], loop);
  VLOG(4) << "After FlattenLoops, ir is:\n" << loop;
}

}  // namespace ir
}  // namespace cinn
