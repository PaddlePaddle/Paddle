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
#include "paddle/infrt/dialect/pd/pass/pd_op_fuse_pass.h"  // NOLINT

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "paddle/infrt/dialect/pd/ir/pd_ops.h"

namespace {
#include "paddle/infrt/dialect/pd/pass/pd_op_fuse.cpp.inc"  // NOLINT

/*
 * PdOpFusePass.
 */
struct PdOpFusePass
    : public mlir::PassWrapper<PdOpFusePass, mlir::FunctionPass> {
 public:
  ::llvm::StringRef getName() const override { return "PdOpFusePass"; }

  llvm::StringRef getArgument() const override { return "pd-op-fuse"; }

  void runOnFunction() override;
};

// Implementation of the PdOpFusePass.
void PdOpFusePass::runOnFunction() {
  ::mlir::RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

}  // namespace

mlir::PassRegistration<PdOpFusePass> infrt_op_fuse_pass;
