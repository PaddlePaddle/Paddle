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

#include "paddle/fluid/pir/transforms/general/auto_layout_simplify_pass.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/common/layout.h"
#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#include "paddle/fluid/pir/dialect/operator/interface/layout_transformation.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pass/utils.h"

namespace {

class RedundantTransposePattern
    : public pir::OpRewritePattern<paddle::dialect::TransposeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::TransposeOp>::OpRewritePattern;

  bool Match(paddle::dialect::TransposeOp op) const override {
    auto before_transpose = op.x().defining_op();
    if (!before_transpose->isa<paddle::dialect::TransposeOp>()) {
      return false;
    }
    if (!(before_transpose->HasAttribute("source"))) return false;
    auto source =
        before_transpose->attribute<pir::StrAttribute>("source").AsString();
    if (source != "auto_layout_pass") return false;
    const auto before_perm_attr =
        before_transpose->attribute<pir::ArrayAttribute>("perm");

    std::vector<int32_t> before_perm;
    for (size_t i = 0; i < before_perm_attr.size(); ++i) {
      auto attr = before_perm_attr.at(i);
      before_perm.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
    }

    const auto after_perm_attr = op.attribute<pir::ArrayAttribute>("perm");
    std::vector<int32_t> after_perm;
    for (size_t i = 0; i < after_perm_attr.size(); ++i) {
      auto attr = after_perm_attr.at(i);
      after_perm.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
    }

    if (before_perm == NCHW2NHWC_ && after_perm == NHWC2NCHW_) return true;
    if (before_perm == NHWC2NCHW_ && after_perm == NCHW2NHWC_) return true;
    return false;
  }
  void Rewrite(paddle::dialect::TransposeOp op,
               pir::PatternRewriter& rewriter) const override {
    auto before_transpose =
        op.x().defining_op()->dyn_cast<paddle::dialect::TransposeOp>();
    rewriter.ReplaceAllUsesWith(op.out(), before_transpose.x());
    rewriter.EraseOp(op);
    if (before_transpose.out().use_empty()) {
      rewriter.EraseOp(before_transpose);
    }
  }

 private:
  const std::vector<int32_t> NCHW2NHWC_ = {0, 2, 3, 1};
  const std::vector<int32_t> NHWC2NCHW_ = {0, 3, 1, 2};
};
class AutoLayoutSimplifyPass : public pir::PatternRewritePass {
 public:
  AutoLayoutSimplifyPass()
      : pir::PatternRewritePass("auto_layout_simplify_pass", 2) {}
  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<RedundantTransposePattern>(context);
    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};
}  // namespace
namespace pir {

std::unique_ptr<Pass> CreateAutoLayoutSimplifyPass() {
  return std::make_unique<AutoLayoutSimplifyPass>();
}

}  // namespace pir

REGISTER_IR_PASS(auto_layout_simplify_pass, AutoLayoutSimplifyPass);
