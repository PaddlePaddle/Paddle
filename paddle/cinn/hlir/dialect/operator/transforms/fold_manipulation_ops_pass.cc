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

#include "paddle/cinn/hlir/dialect/operator/transforms/fold_manipulation_ops_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/refresh_combine_pattern.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"
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

bool RemoveOp(pir::Operation* op,
              pir::PatternRewriter* rewriter,
              bool check_dtype = false) {
  const auto& IsDynamicShape = [](const pir::Value& value) -> bool {
    auto shape_type = value.type().dyn_cast<pir::ShapedTypeInterface>();
    if (shape_type && shape_type.IsDynamicShape()) {
      return true;
    }
    return false;
  };
  const auto& GetDims = [](const pir::Value& value) -> decltype(auto) {
    return value.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
  };

  pir::Value input = op->operand_source(0);
  pir::Value output = op->result(0);
  const auto& IsSameShape = [&]() -> bool {
    const bool has_dynamic_shape =
        IsDynamicShape(input) || IsDynamicShape(output);
    if (has_dynamic_shape) {
      auto& shape_analysis =
          pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());
      auto input_sym_shape =
          GetExprVecFromShape(shape_analysis.GetShapeOrDataForValue(input));
      auto output_sym_shape =
          GetExprVecFromShape(shape_analysis.GetShapeOrDataForValue(output));
      return input_sym_shape == output_sym_shape;
    }
    return GetDims(input) == GetDims(output);
  };

  const auto& IsSameDataType = [&]() -> bool {
    return paddle::dialect::TransToPhiDataType(
               input.type()
                   .dyn_cast<paddle::dialect::DenseTensorType>()
                   .dtype()) ==
           paddle::dialect::TransToPhiDataType(
               output.type()
                   .dyn_cast<paddle::dialect::DenseTensorType>()
                   .dtype());
  };

  const auto CanRemove = [&]() -> bool {
    if (!IsSameShape()) return false;
    if (check_dtype && !IsSameDataType()) return false;
    return true;
  };

  if (CanRemove()) {
    rewriter->ReplaceAllUsesWith(output, input);
    rewriter->EraseOp(op);
    return true;
  }

  return false;
}

template <typename OPTYPE, bool check_dtype = false>
class RemoveUnchangedOpPattern : public pir::OpRewritePattern<OPTYPE> {
 public:
  using pir::OpRewritePattern<OPTYPE>::OpRewritePattern;

  bool MatchAndRewrite(OPTYPE op,
                       pir::PatternRewriter& rewriter) const override {
    return RemoveOp(op, &rewriter, check_dtype);
  }
};

template <typename OPTYPE>
class MergeRedundantOpPattern : public pir::OpRewritePattern<OPTYPE> {
 public:
  using pir::OpRewritePattern<OPTYPE>::OpRewritePattern;

  bool MatchAndRewrite(OPTYPE op,
                       pir::PatternRewriter& rewriter) const override {
    if (auto pre_op = (op->operand_source(0).defining_op())
                          ->template dyn_cast<OPTYPE>()) {
      op->operand(0).set_source(pre_op->operand_source(0));
      if (pre_op->use_empty()) {
        rewriter.EraseOp(pre_op);
      }
      return true;
    }

    return false;
  }
};

class MergeCastOpPattern
    : public pir::OpRewritePattern<paddle::dialect::CastOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::CastOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::CastOp op,
                       pir::PatternRewriter& rewriter) const override {
    const auto& IsFloatType = [&]() -> bool {
      auto in_type = paddle::dialect::TransToPhiDataType(
          op->operand_source(0)
              .type()
              .dyn_cast<paddle::dialect::DenseTensorType>()
              .dtype());
      auto out_type = paddle::dialect::TransToPhiDataType(
          op->result(0)
              .type()
              .dyn_cast<paddle::dialect::DenseTensorType>()
              .dtype());

      return (in_type == phi::DataType::FLOAT16 ||
              in_type == phi::DataType::FLOAT32 ||
              out_type == phi::DataType::FLOAT64) &&
             (out_type == phi::DataType::FLOAT16 ||
              out_type == phi::DataType::FLOAT32 ||
              out_type == phi::DataType::FLOAT64);
    };

    if (!IsFloatType()) {
      return false;
    }

    if (auto pre_op = (op->operand_source(0).defining_op())
                          ->template dyn_cast<paddle::dialect::CastOp>()) {
      op->operand(0).set_source(pre_op->operand_source(0));
      if (pre_op->use_empty()) {
        rewriter.EraseOp(pre_op);
      }
      return true;
    }

    return false;
  }
};

class FoldManipulationOpsPass : public pir::PatternRewritePass {
 public:
  FoldManipulationOpsPass()
      : pir::PatternRewritePass("fold_manipulation_ops_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);

    // remove out_shape equal in_shape ops
    ps.Add<RemoveUnchangedOpPattern<cinn::dialect::ReshapeOp>>(context);
    ps.Add<RemoveUnchangedOpPattern<paddle::dialect::ReshapeOp>>(context);
    ps.Add<RemoveUnchangedOpPattern<cinn::dialect::BroadcastOp>>(context);
    ps.Add<RemoveUnchangedOpPattern<paddle::dialect::ExpandOp>>(context);
    ps.Add<RemoveUnchangedOpPattern<paddle::dialect::AssignOp>>(context);
    ps.Add<RemoveUnchangedOpPattern<cinn::dialect::ConcatOp>>(context);
    ps.Add<RemoveUnchangedOpPattern<paddle::dialect::CastOp, true>>(context);

    // merge redundant ops
    ps.Add<MergeRedundantOpPattern<cinn::dialect::ReshapeOp>>(context);
    ps.Add<MergeRedundantOpPattern<paddle::dialect::ReshapeOp>>(context);
    ps.Add<MergeRedundantOpPattern<cinn::dialect::BroadcastOp>>(context);
    ps.Add<MergeRedundantOpPattern<paddle::dialect::ExpandOp>>(context);
    ps.Add<MergeCastOpPattern>(context);
    ps.Add<RefreshCombineOpPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateFoldManipulationOpsPass() {
  return std::make_unique<FoldManipulationOpsPass>();
}
}  // namespace ir
}  // namespace dialect
}  // namespace cinn

REGISTER_IR_PASS(fold_manipulation_ops_pass,
                 ::cinn::dialect::ir::FoldManipulationOpsPass);
