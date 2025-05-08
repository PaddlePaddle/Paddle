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

#include "paddle/cinn/hlir/dialect/operator/transforms/fold_output_data_derivable_ops_pass.h"

#include <queue>

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

void EraseOpRecursively(pir::Operation* op,
                        pir::PatternRewriter& rewriter) {  // NOLINT
  std::queue<pir::Operation*> ops_queue;
  std::unordered_set<pir::Operation*> erased;
  ops_queue.push(op);
  while (!ops_queue.empty()) {
    auto cur_op = ops_queue.front();
    ops_queue.pop();
    if (erased.count(cur_op)) continue;
    bool no_result_used = true;
    for (const pir::Value& op_result : cur_op->results()) {
      if (op_result.use_count() != 0) {
        no_result_used = false;
        break;
      }
    }
    if (no_result_used) {
      for (const pir::Value& operand_source : cur_op->operands_source()) {
        auto producer_op = operand_source.defining_op();
        if (!producer_op) continue;
        ops_queue.push(producer_op);
      }
      rewriter.EraseOp(cur_op);
      erased.insert(cur_op);
    }
  }
}

class FoldFullAssignValueOpsPattern
    : public pir::OpRewritePattern<paddle::dialect::AssignValue_Op> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::AssignValue_Op>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::AssignValue_Op op,
                       pir::PatternRewriter& rewriter) const override {
    auto* pre_op = op.operand_source(0).defining_op();
    if (pre_op->result(0).use_count() > 1) {
      return false;
    }
    auto new_assign_value_op =
        rewriter.Build<paddle::dialect::AssignValueOp>(op.attributes());
    rewriter.ReplaceAllUsesWith(op.result(0), new_assign_value_op->result(0));
    EraseOpRecursively(op, rewriter);
    return true;
  }
};

template <typename AssignValueOpType>
class FoldAssignValueCastOpsPattern
    : public pir::OpRewritePattern<AssignValueOpType> {
 public:
  using pir::OpRewritePattern<AssignValueOpType>::OpRewritePattern;

  bool MatchAndRewrite(AssignValueOpType op,
                       pir::PatternRewriter& rewriter) const override {
    if (op.result(0).use_count() != 1) return false;
    pir::Operation* next_op = op.result(0).first_use().owner();
    if (!(next_op->isa<paddle::dialect::CastOp>())) return false;

    auto cast_op = next_op->dyn_cast<paddle::dialect::CastOp>();
    pir::AttributeMap attributes = op.attributes();
    attributes["dtype"] = cast_op.attribute("dtype");

    bool is_inplace_op = op.num_operands() == 1;
    pir::Operation* new_assign_value_op;
    if (is_inplace_op) {
      if (op.operand_source(0).use_count() > 1) return false;
      new_assign_value_op = rewriter.Build<paddle::dialect::AssignValue_Op>(
          op->operand_source(0), attributes);
    } else {
      new_assign_value_op =
          rewriter.Build<paddle::dialect::AssignValueOp>(attributes);
    }
    rewriter.ReplaceAllUsesWith(cast_op->result(0),
                                new_assign_value_op->result(0));
    rewriter.EraseOp(cast_op);
    rewriter.EraseOp(op);
    return true;
  }
};

class FoldOutputDataDerivableOps : public pir::RewritePattern {
 private:
  static inline const std::unordered_set<std::string> special_ops = {
      "pd_op.full",
      "pd_op.assign_value",
      "pd_op.assign_value_",
      "pd_op.full_int_array",
      "pd_op.assign",
      "pd_op.memcpy",
  };

 public:
  explicit FoldOutputDataDerivableOps(pir::IrContext* context)
      : RewritePattern(MatchAnyOpTypeTag(),
                       1 /*benefit*/,
                       context,
                       {} /*generated_names*/) {}

  bool MatchAndRewrite(pir::Operation* op,
                       pir::PatternRewriter& rewriter) const override {
    if (special_ops.count(op->name()) || op->num_results() == 0) {
      return false;
    }
    bool non_result_used = true;
    std::vector<std::optional<pir::Operation*>> new_ops;
    for (pir::Value result : op->results()) {
      if (result.use_count() == 0) {
        new_ops.emplace_back(std::nullopt);
        continue;
      }
      non_result_used = false;
      if (!result.type().isa<pir::DenseTensorType>()) return false;
      auto dtype = pir::GetValueDtype(result);
      auto& shape_analysis =
          pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());
      // Get result shape
      auto result_shape = shape_analysis.GetShapeOrDataForValue(result).shape();
      std::vector<int> result_int_shape;
      std::vector<std::int64_t> result_int64_shape;
      int shape_product = 1;
      for (const auto& dim : result_shape) {
        if (dim.isa<std::int64_t>()) {
          auto dim_value = dim.dyn_cast<std::int64_t>();
          result_int_shape.push_back(dim_value);
          result_int64_shape.push_back(dim_value);
          shape_product *= dim_value;
        } else {
          return false;
        }
      }
      // Get result data
      auto opt_result_data =
          shape_analysis.GetShapeOrDataForValue(result).data();
      if (opt_result_data == std::nullopt) return false;
      std::vector<symbol::DimExpr>& result_data = opt_result_data.value();
      std::vector<std::int64_t> result_int_data;
      for (const auto& item : result_data) {
        if (item.isa<std::int64_t>()) {
          result_int_data.push_back(item.dyn_cast<std::int64_t>());
        } else {
          return false;
        }
      }
      if (result_int_data.size() != shape_product) return false;
      // Check whether all elements are same
      bool is_all_data_same = true;
      for (int i = 0; i < shape_product; i++) {
        if (result_int_data[i] != result_int_data[0]) {
          is_all_data_same = false;
          break;
        }
      }
      pir::Operation* new_op;
      if (is_all_data_same && result_int_data.size() > 0) {
        // Build a new full op
        double fill_value = result_int_data[0];
        new_op = rewriter.Build<paddle::dialect::FullOp>(
            result_int64_shape, fill_value, dtype, phi::Place());
      } else {
        // Build a new assign_value op
        std::vector<phi::Scalar> values;
        for (const auto& item : result_int_data) {
          values.emplace_back(item);
        }
        new_op = rewriter.Build<paddle::dialect::AssignValueOp>(
            result_int_shape, dtype, values, phi::Place());
      }
      new_ops.push_back(new_op);
    }
    if (non_result_used) return false;
    for (size_t i = 0; i < new_ops.size(); i++) {
      if (new_ops[i].has_value()) {
        rewriter.ReplaceAllUsesWith(op->result(i),
                                    new_ops[i].value()->result(0));
      }
    }
    EraseOpRecursively(op, rewriter);
    return true;
  }
};

class FoldOutputDataDerivableOpsPass : public pir::PatternRewritePass {
 public:
  FoldOutputDataDerivableOpsPass()
      : pir::PatternRewritePass("fold_output_data_derivable_ops_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<FoldFullAssignValueOpsPattern>(context);
    ps.Add<FoldAssignValueCastOpsPattern<paddle::dialect::AssignValue_Op>>(
        context);
    ps.Add<FoldAssignValueCastOpsPattern<paddle::dialect::AssignValueOp>>(
        context);
    ps.Add<FoldOutputDataDerivableOps>(context);
    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateFoldOutputDataDerivableOpsPass() {
  return std::make_unique<FoldOutputDataDerivableOpsPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
