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

#include "paddle/cinn/hlir/dialect/operator/transforms/shape_ops_fallback_to_phi_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_to_pd_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

class FusionShapeOpsPattern
    : public pir::OpRewritePattern<cinn::dialect::FusionOp> {
 public:
  explicit FusionShapeOpsPattern(::pir::IrContext* context)
      : pir::OpRewritePattern<cinn::dialect::FusionOp>(context) {}

  bool Match(cinn::dialect::FusionOp fusion_op) const override {
    auto& shape_analysis = ::pir::ShapeAnalysisManager::Instance().Get(
        fusion_op->GetParentProgram());
    if (fusion_op.num_results() == 1) {
      const auto& shape =
          shape_analysis.GetShapeOrDataForValue(fusion_op.result(0));
      if (shape.data() && shape.data()->size() <= 9) {
        return true;
      }
    }
    return false;
  }

  void Rewrite(cinn::dialect::FusionOp fusion_op,
               ::pir::PatternRewriter& rewriter) const override {
    ::pir::IrMapping ir_mapping;
    for (auto& op : *fusion_op.block()) {
      if (op.isa<::pir::YieldOp>()) {
        for (uint32_t i = 0; i < op.num_operands(); ++i) {
          rewriter.ReplaceAllUsesWith(
              fusion_op->result(i),
              ir_mapping.Lookup<::pir::Value>(op.operand_source(i)));
        }
        continue;
      }
      for (size_t i = 0; i < op.num_operands(); ++i) {
        if (!ir_mapping.GetMap<::pir::Value>().count(op.operand_source(i))) {
          ir_mapping.Add(op.operand_source(i), op.operand_source(i));
        }
      }
      if (op.dialect()->name() == "cinn_op") {
        auto new_pd_op =
            details::RewriteCinnOpToPdOp(&op, ir_mapping, rewriter);
      } else {
        auto* new_op = op.Clone(ir_mapping, {true, true, true});
        rewriter.Insert(new_op);
      }
    }

    rewriter.EraseOp(fusion_op);
  }
};

class ShapeOpsFallbackToPhiPass : public pir::PatternRewritePass {
 public:
  ShapeOpsFallbackToPhiPass()
      : pir::PatternRewritePass("shape_ops_fallback_to_phi_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<FusionShapeOpsPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

}  // namespace

std::unique_ptr<::pir::Pass> CreateShapeOpsFallbackToPhiPass() {
  return std::make_unique<ShapeOpsFallbackToPhiPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
