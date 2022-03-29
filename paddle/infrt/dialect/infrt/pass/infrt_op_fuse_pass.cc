// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/dialect/infrt/pass/infrt_op_fuse_pass.h"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "paddle/infrt/dialect/infrt/ir/infrt_dialect.h"
#include "paddle/infrt/dialect/pd/ir/pd_ops.h"
namespace {
#include "paddle/infrt/dialect/infrt/pass/infrt_op_fuse.cpp.inc"  // NOLINT

/*
 * infrtOpFusePass.
 */
struct InfrtOpFusePass
    : public mlir::PassWrapper<InfrtOpFusePass, mlir::FunctionPass> {
 public:
  ::llvm::StringRef getName() const override { return "infrtOpFusePass"; }

  llvm::StringRef getArgument() const override { return "infrt-op-fuse"; }

  void runOnFunction() override;
};

// Implementation of the InfrtOpFusePass.
void InfrtOpFusePass::runOnFunction() {
  ::mlir::RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  // Fuse infrt.return Operation
  auto terminator_op = getFunction().front().getTerminator();
  if (nullptr == terminator_op) return;
  for (auto operand : terminator_op->getOperands()) {
    auto *op1 = operand.getDefiningOp();
    auto cvt_op = ::llvm::dyn_cast<::infrt::TensorCastOp>(op1);
    if (!cvt_op) continue;
    mlir::Value value = cvt_op.input();
    operand.replaceAllUsesWith(value);
    cvt_op.erase();
  }
}

}  // namespace

std::unique_ptr<mlir::Pass> infrt::createInfrtOpFusePass() {
  return std::make_unique<InfrtOpFusePass>();
}

mlir::PassRegistration<InfrtOpFusePass> infrt_op_fuse_pass;
