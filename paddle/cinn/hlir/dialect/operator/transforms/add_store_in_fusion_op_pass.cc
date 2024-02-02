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

#include "paddle/cinn/hlir/dialect/operator/transforms/add_store_in_fusion_op_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builtin_type_interfaces.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

class AddStoreInFusionOpPattern
    : public pir::OpRewritePattern<cinn::dialect::FusionOp> {
 public:
  using pir::OpRewritePattern<cinn::dialect::FusionOp>::OpRewritePattern;

  bool MatchAndRewrite(cinn::dialect::FusionOp op,
                       pir::PatternRewriter& rewriter) const override {
    auto yield_op = op.GetOperators().back();

    for (auto i = 0; i < yield_op->num_operands(); ++i) {
      auto new_full = rewriter.Build<cinn::dialect::StoreOp>(
          yield_op->operand_source(i), yield_op->operand_source(i).type());

      yield_op->operand(i).set_source(new_full.result(0));
    }

    return true;
  }
};

class AddStoreInFusionOpPass : public pir::Pass {
 public:
  AddStoreInFusionOpPass()
      : pir::Pass("cinn_dynamic_reshape_op_pass", /*opt_level=*/1) {}

  bool Initialize(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<AddStoreInFusionOpPattern>(context);

    patterns_ = pir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(pir::Operation* op) override {
    pir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 10;
    for (uint32_t i = 0; i < op->num_regions(); ++i) {
      for (auto& block : op->region(i)) {
        for (auto& op : block) {
          if (op.isa<cinn::dialect::FusionOp>()) {
            auto [_, num_rewrites] =
                pir::ApplyPatternsGreedily(&op, patterns_, cfg);
            AddStatistics(num_rewrites);
          }
        }
      }
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }

 private:
  pir::FrozenRewritePatternSet patterns_;
};

std::unique_ptr<pir::Pass> CreateAddStoreInFusionOpPass() {
  return std::make_unique<AddStoreInFusionOpPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
