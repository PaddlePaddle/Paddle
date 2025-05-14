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

#include "paddle/ap/include/paddle/pass/fallback_fusion_op_to_phi_pass.h"
#include "paddle/ap/include/paddle/hlir/manual_op.h"

#include "paddle/ap/include/axpr/abstract_list.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/atomic.h"
#include "paddle/ap/include/axpr/builtin_frame_util.h"
#include "paddle/ap/include/axpr/builtin_serializable_attr_map_to_axpr_helper.h"
#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/paddle/builtin_frame_util.h"
#include "paddle/ap/include/paddle/pir/ap_pir_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_mapping.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace ap::paddle {

namespace adt = ap::adt;

namespace {

template <bool parent_must_be_fusion_op>
class FallbackFusionOpToPhiPattern
    : public pir::OpRewritePattern<::cinn::dialect::FusionOp> {
 public:
  using pir::OpRewritePattern<::cinn::dialect::FusionOp>::OpRewritePattern;

  bool MatchAndRewrite(::cinn::dialect::FusionOp fusion_op,
                       pir::PatternRewriter& rewriter) const override {
    const auto& ret = TryMatchAndRewrite(fusion_op, &rewriter);
    PADDLE_ENFORCE_EQ(
        ret.HasError(),
        false,
        phi::errors::Fatal(
            "FallbackFusionOpToPhiPattern::MatchAndRewrite failed. "
            "\nTraceback (most recent call "
            "last):\n%s\n%s: %s. ",
            ret.GetError().CallStackToString(),
            ret.GetError().class_name(),
            ret.GetError().msg()));
    return ret.GetOkValue();
  }

  adt::Result<bool> TryMatchAndRewrite(::cinn::dialect::FusionOp fusion_op,
                                       pir::PatternRewriter* rewriter) const {
    auto* mut_block = fusion_op->GetParent();
    if constexpr (parent_must_be_fusion_op) {
      if (!fusion_op->GetParentOp()->isa<::cinn::dialect::FusionOp>())
        return false;
    }
    pir::IrMapping ir_mapping{};
    for (pir::Value free_value : pir::GetUsedExternalValue(*fusion_op)) {
      ir_mapping.Add(free_value, free_value);
    }
    std::vector<pir::Value> yield_inputs{};
    {
      yield_inputs.reserve(fusion_op->num_results());
      auto clone_options = pir::CloneOptions(true, true, true);
      for (auto& op : *fusion_op.block()) {
        if (op.isa<pir::YieldOp>()) {
          yield_inputs = op.operands_source();
        } else {
          rewriter->Insert(op.Clone(ir_mapping, clone_options));
        }
      }
    }
    for (int i = 0; i < fusion_op->num_results(); ++i) {
      rewriter->ReplaceAllUsesWith(fusion_op->result(i),
                                   ir_mapping.Lookup(yield_inputs.at(i)));
    }
    rewriter->EraseOp(fusion_op);
    return true;
  }
};

template <bool parent_must_be_fusion_op>
class FallbackFusionOpToPhiPass : public pir::PatternRewritePass {
 public:
  FallbackFusionOpToPhiPass()
      : pir::PatternRewritePass("fallback_fusion_op_to_phi_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<FallbackFusionOpToPhiPattern<parent_must_be_fusion_op>>(context);
    return ps;
  }
};

}  // namespace

std::unique_ptr<::pir::Pass> CreateFallbackFusionOpToPhiPass() {
  return std::make_unique<
      FallbackFusionOpToPhiPass</*parent_must_be_fusion_op=*/false>>();
}

std::unique_ptr<::pir::Pass> CreateFallbackNestedFusionOpToPhiPass() {
  return std::make_unique<
      FallbackFusionOpToPhiPass</*parent_must_be_fusion_op=*/true>>();
}

}  // namespace ap::paddle
