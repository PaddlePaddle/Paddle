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

#include "paddle/fluid/pir/transforms/general/delete_assert_op_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace pir {

class DeleteAssertOpPattern
    : public OpRewritePattern<paddle::dialect::AssertOp> {
 public:
  using OpRewritePattern<paddle::dialect::AssertOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::AssertOp op,
                       PatternRewriter& rewriter) const override {  // NOLINT
    auto data_defining_op = op.data().defining_op();
    rewriter.EraseOp(op);

    if (data_defining_op && data_defining_op->isa<CombineOp>() &&
        op.data().use_empty()) {
      rewriter.EraseOp(data_defining_op);
    }
    return true;
  }
};

class DeleteAssertOpPass : public PatternRewritePass {
 public:
  DeleteAssertOpPass() : PatternRewritePass("delete_assert_op_pass", 1) {}

  RewritePatternSet InitializePatterns(IrContext* context) override {
    RewritePatternSet ps(context);
    ps.Add<DeleteAssertOpPattern>(context);
    return ps;
  }

 private:
  FrozenRewritePatternSet patterns_;
};

std::unique_ptr<Pass> CreateDeleteAssertOpPass() {
  return std::make_unique<DeleteAssertOpPass>();
}

}  // namespace pir

REGISTER_IR_PASS(delete_assert_op_pass, pir::DeleteAssertOpPass);
