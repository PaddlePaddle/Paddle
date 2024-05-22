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

#include "paddle/cinn/hlir/dialect/operator/transforms/accuracy_check_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_to_pd_util.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn::dialect::ir {

class AddAccuracyCheckPattern
    : public pir::OpRewritePattern<cinn::dialect::FusionOp> {
 public:
  using pir::OpRewritePattern<cinn::dialect::FusionOp>::OpRewritePattern;

  bool MatchAndRewrite(cinn::dialect::FusionOp fusion_op,
                       pir::PatternRewriter& rewriter) const override {
    const auto op_list = fusion_op.GetOperators();

    const auto group_info = fusion_op.attribute("group_info")
                                .dyn_cast<cinn::dialect::GroupInfoAttribute>()
                                .data();
    const auto& fn_name = group_info.fn_name;

    ::pir::IrMapping ir_mapping;
    ::pir::CloneOptions clone_options(/*clone_regions=*/false,
                                      /*clone_operands=*/true,
                                      /*clone_successors=*/false);
    ::pir::IrContext* ctx = ::pir::IrContext::Instance();
    ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
    ::pir::Builder builder = ::pir::Builder(ctx, fusion_op->GetParent());
    builder.set_insertion_point(fusion_op);

    const auto& InsertAccuaryCheckOp = [&](::pir::Operation* op) -> void {
      for (size_t i = 0; i < op->num_operands(); ++i) {
        rewriter.Build<paddle::dialect::AccuracyCheckOp>(
            fusion_op.result(i),
            ir_mapping.Lookup(op->operand_source(i)),
            fn_name,
            i);
      }
    };

    const auto& ConvertCinnOpToPdOp = [&](::pir::Operation* op) -> void {
      rewriter.SetInsertionPointAfter(fusion_op);
      for (size_t i = 0; i < op->num_operands(); ++i) {
        if (!ir_mapping.GetMap<pir::Value>().count(op->operand_source(i))) {
          ir_mapping.Add(op->operand_source(i), op->operand_source(i));
        }
      }
      pir::Operation* pd_op =
          cinn::dialect::details::RewriteCinnOpToPdOp(op, ir_mapping, builder);
    };

    const auto& ClonePdOp = [&](::pir::Operation* op) -> void {
      for (size_t i = 0; i < op->num_operands(); ++i) {
        if (!ir_mapping.GetMap<pir::Value>().count(op->operand_source(i))) {
          ir_mapping.Add(op->operand_source(i), op->operand_source(i));
        }
      }
      auto new_op = op->Clone(ir_mapping, clone_options);
      rewriter.Insert(new_op);
      rewriter.SetInsertionPointAfter(new_op);
    };

    for (auto& op : op_list) {
      if (op->isa<::pir::YieldOp>()) {
        InsertAccuaryCheckOp(op);
      } else if (op->dialect()->name() == "cinn_op") {
        ConvertCinnOpToPdOp(op);
      } else {
        ClonePdOp(op);
      }
    }
    return true;
  }
};

class AccuarcyCheckPass : public pir::Pass {
 public:
  AccuarcyCheckPass() : pir::Pass("accuracy_check_pass", /*opt_level=*/3) {}

  bool Initialize(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<AddAccuracyCheckPattern>(context);

    patterns_ = pir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(pir::Operation* op) override {
    int64_t num_ops{0};
    for (uint32_t i = 0; i < op->num_regions(); ++i) {
      auto& region = op->region(i);
      for (auto& block : region) {
        num_ops += block.size();
      }
    }
    pir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 1;
    auto [_, num_rewrites] = pir::ApplyPatternsGreedily(op, patterns_, cfg);
    AddStatistics(num_rewrites, num_ops);
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }

 private:
  pir::FrozenRewritePatternSet patterns_;
};

std::unique_ptr<pir::Pass> CreateAccuarcyCheckPass() {
  return std::make_unique<AccuarcyCheckPass>();
}

}  // namespace cinn::dialect::ir
