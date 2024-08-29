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

#include "paddle/cinn/hlir/dialect/operator/transforms/fusion_fallback_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_to_pd_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/split_generate_shape_into_shape_ops_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

class FusionOpsPattern : public pir::OpRewritePattern<cinn::dialect::FusionOp> {
 public:
  explicit FusionOpsPattern(::pir::IrContext* context)
      : pir::OpRewritePattern<cinn::dialect::FusionOp>(context) {}

  bool MatchAndRewrite(cinn::dialect::FusionOp fusion_op,
                       ::pir::PatternRewriter& rewriter) const override {
    ::pir::IrMapping ir_mapping;

    const auto& ConvertCinnOpToPdOp = [&](::pir::Operation* op) -> void {
      if (op->isa<cinn::dialect::GenerateShapeOp>()) {
        auto cloned_op = op->Clone(ir_mapping, {false, true, false});
        rewriter.Insert(cloned_op);
        auto cinn_op = cloned_op->dyn_cast<cinn::dialect::GenerateShapeOp>();
        std::optional<pir::Value> out_replacement =
            details::GetOutReplacement(cinn_op, &rewriter);
        if (!out_replacement.has_value()) return;

        rewriter.ReplaceAllUsesWith(cloned_op->result(0),
                                    out_replacement.value());
        PADDLE_ENFORCE_EQ(cloned_op->use_empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "cinn_op.generate_shape op shouldn't "
                              "be used outside fusion block."));
        rewriter.EraseOp(cloned_op);
        ir_mapping.Add(op->result(0), out_replacement.value());
        rewriter.SetInsertionPointAfter(out_replacement.value().defining_op());
        return;
      }
      if (op->isa<cinn::dialect::YieldStoreOp>()) {
        VLOG(6) << "skip yield_store op";
        ir_mapping.Add(op->result(0), ir_mapping.Lookup(op->operand_source(0)));
        return;
      }
      pir::Operation* pd_op =
          cinn::dialect::details::RewriteCinnOpToPdOp(op, ir_mapping, rewriter);
      rewriter.SetInsertionPointAfter(pd_op);
    };

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
        ConvertCinnOpToPdOp(&op);
      } else {
        auto* new_op = op.Clone(ir_mapping, {true, true, true});
        rewriter.Insert(new_op);
      }
    }

    rewriter.EraseOp(fusion_op);
    return true;
  }
};

class FusionFallbackPass : public pir::PatternRewritePass {
 public:
  FusionFallbackPass() : pir::PatternRewritePass("fusion_fallback_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<FusionOpsPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

}  // namespace

std::unique_ptr<::pir::Pass> CreateFusionFallbackPass() {
  return std::make_unique<FusionFallbackPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
