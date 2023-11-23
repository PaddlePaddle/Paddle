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
#include "paddle/cinn/ir/dy_schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"

namespace cinn {
namespace ir {

struct FindBlockParent : public ir::IRMutator<> {
 public:
  explicit FindBlockParent(const std::string& block_name)
      : block_name_(block_name) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::Block* expr, Expr* op) override {
    if (target_) return;
    for (auto& stmt : expr->stmts) {
      if (stmt.As<ir::ScheduleBlockRealize>()) {
        if (stmt.As<ir::ScheduleBlockRealize>()
                ->schedule_block.As<ir::ScheduleBlock>()
                ->name == block_name_) {
          target_ = op;
          return;
        }
      }
    }
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::For* expr, Expr* op) override {
    if (target_) return;
    if (expr->body.As<ir::ScheduleBlockRealize>()) {
      if (expr->body.As<ir::ScheduleBlockRealize>()
              ->schedule_block.As<ir::ScheduleBlock>()
              ->name == block_name_) {
        target_ = op;
        return;
      }
    }
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::ScheduleBlock* expr, Expr* op) override {
    if (target_) return;
    if (expr->body.As<ir::ScheduleBlockRealize>()) {
      if (expr->body.As<ir::ScheduleBlockRealize>()
              ->schedule_block.As<ir::ScheduleBlock>()
              ->name == block_name_) {
        target_ = op;
        return;
      }
    }
    IRMutator::Visit(expr, op);
  }

  std::string block_name_;

 public:
  ir::Expr* target_{nullptr};
};

std::vector<Expr> DyScheduleImpl::Split(const Expr& loop,
                                        const std::vector<int>& factors) {
  CHECK(loop.As<ir::For>())
      << "Expr param of Split must be For node! Please check.";
  auto* for_node = loop.As<ir::For>();
  CHECK(common::is_zero(for_node->min))
      << "The For node must start with 0! Please check.";
  CHECK(!factors.empty())
      << "The factors param of Split should not be empty! Please check.";

  Expr tot_extent = for_node->extent;

  VLOG(3) << "Try Split loop from (" << for_node->loop_var->name << ", 0, "
          << tot_extent << ") to (" << cinn::utils::Join(factors, ", ")
          << ") at loop:\n"
          << loop;

  bool is_positive = true;
  std::vector<Expr> process_factors;
  std::for_each(factors.begin(), factors.end(), [&](int factor) {
    process_factors.push_back(Expr(factor));
    is_positive = is_positive && (factor > 0);
  });
  CHECK(is_positive)
      << "The params in factors of Split on dynamic shape should be positive\n";

  // No Longer need to check factor is valid or not because they should be
  // checked outside in group schedule
  // // CINN_IR_SCHEDULE_BEGIN();
  // processed_factors = ValidateFactors(factors, tot_extent,
  // this->module_expr_); CINN_IR_SCHEDULE_END(this->err_msg_level_);

  auto prod_size = std::accumulate(factors.begin(), factors.end(), Expr(1));
  process_factors.insert(process_factors.begin(),
                         tot_extent / prod_size + Expr(1));

  std::vector<Var> new_loop_vars;
  new_loop_vars.push_back(Var(common::UniqName(for_node->loop_var->name)));
  Expr substitute_value(1);
  for (int i = 0; i < process_factors.size(); ++i) {
    Var temp_var(common::UniqName(for_node->loop_var->name));
    substitute_value = Expr(temp_var) + substitute_value * processed_factors[i];
    new_loop_vars.push_back(temp_var);
  }

  Expr new_node = ir::ir_utils::IRCopy(for_node->body);
  ReplaceExpr(&new_node, {for_node->loop_var}, {substitute_value});
  std::vector<Expr> splited_loops;
  splited_loops.resize(process_factors.size());

  new_node =
      IfThenElse::Make(LT::Make(substitute_value, for_node->extent), new_node);

  for (int i = process_factors.size() - 1; i >= 0; i--) {
    if (!new_node.As<ir::Block>()) new_node = Block::Make({new_node});
    new_node = For::Make(new_loop_vars[i],
                         Expr(0),
                         processed_factors[i],
                         for_node->for_type(),
                         for_node->device_api,
                         new_node);
    splited_loops[i] = new_node;
  }

  this->Replace(loop, new_node);
  VLOG(3) << "After Split, ir is:\n" << splited_loops.at(0);
  return splited_loops;
}

Expr DyScheduleImpl::Fuse(const std::vector<Expr>& loops) {
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
      CHECK_EQ(for_nodes.back()->body.As<ir::Block>()->stmts.size(), 1U)
          << "The Block'size of for node is not 1!";
      CHECK_EQ(for_nodes.back()->body.As<ir::Block>()->stmts[0], it_loop)
          << "The For nodes in loops param of Fuse must be adjacent! Please "
             "check.";
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

  // current AutoSimplify doesn't support symbolic_dim
  // fused_extent = common::AutoSimplify(fused_extent);

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

Expr DyScheduleImpl::Fuse(const std::string& block_name,
                          const std::vector<int>& loops_index) {
  std::vector<Expr> all_loops = this->GetLoops(block_name);
  std::vector<Expr> loops_expr;
  loops_expr.reserve(loops_index.size());
  for (int i = 0; i < loops_index.size(); ++i) {
    if (i > 0)
      CHECK_EQ(loops_index[i - 1] + 1, loops_index[i])
          << "Loops index in Fuse shoule be continuous!";
  }
  for (int i : loops_index) {
    CHECK_LT(i, (int)all_loops.size())
        << "The loop index in Fuse should be less than total loop's number.";
    CHECK_GE(i, 0) << "The loop index in Fuse should be >= 0.";
    loops_expr.emplace_back(all_loops[i]);
  }
  return this->Fuse(loops_expr);
}

Expr DyScheduleImpl::Fuse(const Expr& block,
                          const std::vector<int>& loops_index) {
  std::vector<Expr> all_loops = this->GetLoops(block);
  std::vector<Expr> loops_expr;
  loops_expr.reserve(loops_index.size());
  for (int i = 0; i < loops_index.size(); ++i) {
    if (i > 0)
      CHECK_EQ(loops_index[i - 1] + 1, loops_index[i])
          << "Loops index in Fuse shoule be continuous!";
  }
  for (int i : loops_index) {
    CHECK_LT(i, (int)all_loops.size())
        << "The loop index in Fuse should be less than total loop's number.";
    CHECK_GE(i, 0) << "The loop index in Fuse should be >= 0.";
    loops_expr.emplace_back(all_loops[i]);
  }
  return this->Fuse(loops_expr);
}

Expr DyScheduleImpl::Reorder(const std::vector<Expr>& loops) {
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

Expr DyScheduleImpl::Reorder(const std::string& block_name,
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

Expr DyScheduleImpl::Reorder(const Expr& block,
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

Expr DyScheduleImpl::AddUnitLoop(const Expr& block) const {
  auto exprs = module_expr_.GetExprs();
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(block.As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>());
  std::string block_name = block.As<ir::ScheduleBlockRealize>()
                               ->schedule_block.As<ir::ScheduleBlock>()
                               ->name;

  FindBlockParent visitor(block_name);
  for (auto expr : exprs) {
    visitor(&expr);
    if (visitor.target_) {
      break;
    }
  }

  CHECK(visitor.target_) << ", block name : " << block_name << "\n" << exprs;
  if (visitor.target_->As<ir::Block>()) {
    for (auto& stmt : visitor.target_->As<ir::Block>()->stmts) {
      if (stmt.As<ir::ScheduleBlockRealize>()) {
        if (stmt.As<ir::ScheduleBlockRealize>()
                ->schedule_block.As<ir::ScheduleBlock>()
                ->name == block_name) {
          auto block = ir::Block::Make({GetBlock(block_name)});
          auto loop = ir::For::Make(ir::Var(common::UniqName("ix")),
                                    ir::Expr(0),
                                    ir::Expr(1),
                                    ir::ForType::Serial,
                                    ir::DeviceAPI::UNK,
                                    block);
          stmt = loop;
          return loop;
        }
      }
    }
  } else if (visitor.target_->As<ir::For>()) {
    auto block = ir::Block::Make({visitor.target_->As<ir::For>()->body});
    auto loop = ir::For::Make(ir::Var(common::UniqName("ix")),
                              ir::Expr(0),
                              ir::Expr(1),
                              ir::ForType::Serial,
                              ir::DeviceAPI::UNK,
                              block);
    visitor.target_->As<ir::For>()->body = loop;
    return loop;
  } else if (visitor.target_->As<ir::ScheduleBlock>()) {
    auto block =
        ir::Block::Make({visitor.target_->As<ir::ScheduleBlock>()->body});
    auto loop = ir::For::Make(ir::Var(common::UniqName("ix")),
                              ir::Expr(0),
                              ir::Expr(1),
                              ir::ForType::Serial,
                              ir::DeviceAPI::UNK,
                              block);
    visitor.target_->As<ir::ScheduleBlock>()->body = loop;
    return loop;
  } else {
    LOG(FATAL) << "Can't find block's parent!";
  }
  LOG(FATAL) << "Shouldn't reach code here in AddUnitLoop";
  return Expr{nullptr};
}

void DyScheduleImpl::FlattenLoops(const std::vector<Expr>& loops,
                                  const bool force_flat) {
  std::for_each(loops.begin(), loops.end(), [](Expr& loop) {
    if (!loop.As<For>()->extent.is_constant())
      LOG(FATAL) << "Loop with extent which is variable can't be flatten\n";
  })
}

}  // namespace ir
}  // namespace cinn
