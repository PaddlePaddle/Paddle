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

#include "paddle/cinn/hlir/dialect/operator/transforms/remove_assign_out_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/refresh_combine_pattern.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_match_context.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {
using paddle::dialect::details::GetExprVecFromShape;

bool CanRemove(pir::Block::ConstIterator assign_out_op,
               const pir::Block& body) {
  pir::Block::ConstIterator it = assign_out_op;
  it++;

  std::unordered_set<::pir::Operation*> next_ops;
  for (; it != body.end(); ++it) {
    next_ops.insert(it);
  }

  auto out_value = assign_out_op->operand_source(1);

  for (auto use_it = out_value.use_begin(); use_it != out_value.use_end();
       ++use_it) {
    if (next_ops.count(use_it->owner())) {
      return false;
    }
  }

  return true;
}

class RemoveAssignOutPattern
    : public pir::OpRewritePattern<paddle::dialect::WhileOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::WhileOp>::OpRewritePattern;

  bool Match(paddle::dialect::WhileOp op) const override {
    auto& body_block = op.body();

    auto block_args = op.block_args();

    std::unordered_set<::pir::Value> args_set(block_args.begin(),
                                              block_args.end());

    for (auto op : body_block.ops()) {
      if (op->isa<paddle::dialect::AssignOut_Op>() &&
          args_set.count(op->operand_source(1))) {
        return true;
      }
    }

    return false;
  }

  void Rewrite(paddle::dialect::WhileOp op,
               pir::PatternRewriter& rewriter) const override {
    auto& body_block = op.body();

    std::vector<pir::Block::ConstIterator> check_list;

    for (pir::Block::ConstIterator it = body_block.begin();
         it != body_block.end();
         ++it) {
      if (it->isa<paddle::dialect::AssignOut_Op>()) {
        check_list.push_back(it);
      }
    }

    std::reverse(check_list.begin(), check_list.end());
    for (auto check_it : check_list) {
      if (CanRemove(check_it, body_block)) {
        // replace user first
        rewriter.ReplaceAllUsesWith(check_it->result(0),
                                    check_it->operand_source(0));

        body_block.erase(check_it);
      }
    }
  }
};

class RemoveAssignOutPass : public pir::PatternRewritePass {
 public:
  RemoveAssignOutPass()
      : pir::PatternRewritePass("remove_assign_out_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);

    // remove assign out
    ps.Add<RemoveAssignOutPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateRemoveAssignOutPass() {
  return std::make_unique<RemoveAssignOutPass>();
}
}  // namespace ir
}  // namespace dialect
}  // namespace cinn
