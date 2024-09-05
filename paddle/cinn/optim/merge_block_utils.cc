// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/merge_block_utils.h"

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace optim {

struct ForVarExtent {
  ir::Var loop_var;
  ir::Expr extent;
};

struct ForInfoChecker : public ir::IRMutator<Expr*> {
 public:
  ForInfoChecker(const ir::ScheduleBlock* block1,
                 const ir::ScheduleBlock* block2)
      : block1_(block1), block2_(block2) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  bool IsBlockForEqual() {
    auto ForVarExtentEqual =
        [&](const std::vector<ForVarExtent>& for_var_extent1,
            const std::vector<ForVarExtent>& for_var_extent2) -> bool {
      if (for_var_extent1.size() != for_var_extent2.size()) return false;

      for (size_t i = 0; i < for_var_extent1.size(); ++i) {
        const ir::Expr lhs = for_var_extent1[i].extent;
        const ir::Expr rhs = for_var_extent2[i].extent;
        if (cinn::common::AutoSimplify(ir::Sub::Make(lhs, rhs)) !=
            ir::Expr(0)) {
          return false;
        }
      }
      return true;
    };

    if (block_name_to_for_var_extents_.size() <= 1) {
      return false;
    }
    if (block_name_to_for_var_extents_.count(block1_->name) == 0 ||
        block_name_to_for_var_extents_.count(block2_->name) == 0) {
      return false;
    }

    return ForVarExtentEqual(block_name_to_for_var_extents_[block1_->name],
                             block_name_to_for_var_extents_[block2_->name]);
  }

 private:
  void Visit(const ir::For* op, ir::Expr* expr) {
    auto* node = expr->As<ir::For>();
    for_var_extents_.push_back({node->loop_var, node->extent});
    ir::IRMutator<>::Visit(op, expr);
    for_var_extents_.pop_back();
  }

  void Visit(const ir::ScheduleBlock* op, ir::Expr* expr) {
    auto* node = expr->As<ir::ScheduleBlock>();
    if (node->name == block1_->name || node->name == block2_->name) {
      block_name_to_for_var_extents_[node->name] = for_var_extents_;
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  std::vector<ForVarExtent> for_var_extents_;
  std::unordered_map<std::string, std::vector<ForVarExtent>>
      block_name_to_for_var_extents_;

  const ir::ScheduleBlock* block1_;
  const ir::ScheduleBlock* block2_;
};

bool CanMergeBlocks(ir::Expr* source,
                    const ir::ScheduleBlock* block1,
                    const ir::ScheduleBlock* block2) {
  ForInfoChecker for_info_checker(block1, block2);
  for_info_checker(source);
  return for_info_checker.IsBlockForEqual();
}

}  // namespace optim
}  // namespace cinn
