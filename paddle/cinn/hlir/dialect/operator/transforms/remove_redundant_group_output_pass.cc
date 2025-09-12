// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/dialect/operator/transforms/remove_redundant_group_output_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builtin_type_interfaces.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

class RemoveRedundantGroupOutputPattern
    : public pir::OpRewritePattern<cinn::dialect::GroupOp> {
 public:
  explicit RemoveRedundantGroupOutputPattern(::pir::IrContext* context)
      : pir::OpRewritePattern<cinn::dialect::GroupOp>(context) {}

  bool MatchAndRewrite(cinn::dialect::GroupOp group_op,
                       pir::PatternRewriter& rewriter) const override {
    // Detect external inputs of the group op, if it's directly used by the
    // yield op, remove it from the group op's output.
    const auto& input_args = pir::GetUsedExternalValue(*group_op.block());
    const std::unordered_set<pir::Value> inputs_set = {input_args.begin(),
                                                       input_args.end()};
    const auto& yield_op = group_op.block()->back();
    std::vector<pir::Value> pruned_outputs;
    std::unordered_set<uint32_t> redundant_out_indices;
    for (uint32_t i = 0; i < yield_op.num_operands(); ++i) {
      if (inputs_set.count(yield_op.operand_source(i)) > 0) {
        redundant_out_indices.insert(i);
      } else {
        pruned_outputs.push_back(yield_op.operand_source(i));
      }
    }
    if (redundant_out_indices.empty()) {
      VLOG(7) << "No redundant output in group op, skip.";
      return false;
    }

    // Create new group op and yield op and move other ops into the new group
    // op.
    std::vector<pir::Type> new_out_types =
        [](const std::vector<pir::Value>& values) {
          std::vector<pir::Type> types;
          for (auto& value : values) {
            types.push_back(value.type());
          }
          return types;
        }(pruned_outputs);
    auto new_group_op = rewriter.Build<cinn::dialect::GroupOp>(new_out_types);
    const std::vector<pir::Operation*> ops_to_move = [](pir::Block* block) {
      std::vector<pir::Operation*> ops;
      for (auto& op : *block) {
        if (op.isa<pir::YieldOp>()) continue;
        ops.push_back(&op);
      }
      return ops;
    }(group_op.block());
    for (auto& op : ops_to_move) {
      op->MoveTo(new_group_op.block(), new_group_op.block()->end());
    }
    rewriter.SetInsertionPointToBlockEnd(new_group_op.block());
    rewriter.Build<pir::YieldOp>(pruned_outputs);

    // Replace the group op outputs with the new group op outputs and
    // external inputs.
    uint32_t new_out_idx = 0;
    for (uint32_t i = 0; i < group_op.num_results(); ++i) {
      if (redundant_out_indices.count(i) > 0) {
        rewriter.ReplaceAllUsesWith(group_op.result(i),
                                    yield_op.operand_source(i));
      } else {
        rewriter.ReplaceAllUsesWith(group_op.result(i),
                                    new_group_op.result(new_out_idx));
        new_out_idx += 1;
      }
    }
    rewriter.EraseOp(group_op);
    VLOG(7) << "Remove " << redundant_out_indices.size()
            << " redundant outputs from group op.";
    return true;
  }
};

class RemoveRedundantGroupOutputPass : public pir::PatternRewritePass {
 public:
  RemoveRedundantGroupOutputPass()
      : pir::PatternRewritePass("remove_redundant_group_output_pass",
                                /*opt_level=*/1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);

    ps.Add<RemoveRedundantGroupOutputPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    if (op->isa<cinn::dialect::GroupOp>()) {
      return false;
    }
    return op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateRemoveRedundantGroupOutputPass() {
  return std::make_unique<RemoveRedundantGroupOutputPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
