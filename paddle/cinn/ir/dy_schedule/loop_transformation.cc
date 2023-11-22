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

namespace cinn {
namespace ir {

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
  process_factors.insert(process_factors.begin(), tot_extent / prod_size);

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
  CINN_NOT_IMPLEMENTED;
}

Expr DyScheduleImpl::Fuse(const std::string& block_name,
                          const std::vector<int>& loops_index) {
  CINN_NOT_IMPLEMENTED;
}

Expr DyScheduleImpl::Fuse(const Expr& block,
                          const std::vector<int>& loops_index) {
  CINN_NOT_IMPLEMENTED;
}

Expr DyScheduleImpl::Reorder(const std::vector<Expr>& loops) {
  CINN_NOT_IMPLEMENTED;
}

Expr DyScheduleImpl::Reorder(const std::string& block_name,
                             const std::vector<int>& loops_index) {
  CINN_NOT_IMPLEMENTED;
}

Expr DyScheduleImpl::Reorder(const Expr& block,
                             const std::vector<int>& loops_index) {
  CINN_NOT_IMPLEMENTED;
}

Expr DyScheduleImpl::AddUnitLoop(const Expr& block) const {
  CINN_NOT_IMPLEMENTED;
}

void DyScheduleImpl::FlattenLoops(const std::vector<Expr>& loops,
                                  const bool force_flat) {
  CINN_NOT_IMPLEMENTED;
}

}  // namespace ir
}  // namespace cinn
