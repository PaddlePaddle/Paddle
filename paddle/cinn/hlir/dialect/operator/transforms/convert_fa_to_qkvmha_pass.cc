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

#include "paddle/cinn/hlir/dialect/operator/transforms/convert_fa_to_qkvmha_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

class ConvertFA2QKVMHAPattern
    : public pir::OpRewritePattern<paddle::dialect::FlashAttnOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::FlashAttnOp>::OpRewritePattern;

  bool Match(paddle::dialect::FlashAttnOp op) const override {
    bool is_test =
        op->attribute("is_test").dyn_cast<pir::BoolAttribute>().data();
    if (!is_test) {
      return false;
    }
    auto& shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());

    auto in_shape =
        shape_analysis.GetShapeOrDataForValue(op->operand_source(0));

    auto dim1 = in_shape.shape()[1];

    if (dim1.isa<int64_t>() &&
        (dim1.dyn_cast<int64_t>() == static_cast<int64_t>(1))) {
      return true;
    }

    return false;
  }

  void Rewrite(paddle::dialect::FlashAttnOp op,
               pir::PatternRewriter& rewriter) const override {
    auto q = op->operand_source(0);
    auto k = op->operand_source(1);
    auto v = op->operand_source(2);

    pir::Value attn_mask;
    auto fa =
        rewriter.Build<paddle::dialect::QkvUnpackMhaOp>(q, k, v, attn_mask);

    rewriter.ReplaceAllUsesWith(op->result(0), fa.result(0));
    rewriter.EraseOp(op);
  }
};

class ConvertFA2QKVMHAPass : public pir::PatternRewritePass {
 public:
  ConvertFA2QKVMHAPass() : pir::PatternRewritePass("convert_FA_to_QKVMHA", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<ConvertFA2QKVMHAPattern>(context);
    return ps;
  }
};

std::unique_ptr<pir::Pass> CreateConvertFA2QKVMHAPass() {
  return std::make_unique<ConvertFA2QKVMHAPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
