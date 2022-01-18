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

#include "paddle/infrt/dialect/pd_test_op_kernel_mapping_pass.h"

#include <glog/logging.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

#include "paddle/infrt/dialect/infrt_base.h"
#include "paddle/infrt/dialect/pd_ops.h"

namespace infrt {
namespace pd {
#include "paddle/infrt/dialect/rewrite.hpp.inc"
}  // namespace pd
}  // namespace infrt

namespace infrt {

class NaivePatternRewriter : public mlir::PatternRewriter {
 public:
  explicit NaivePatternRewriter(mlir::MLIRContext* ctx)
      : mlir::PatternRewriter(ctx) {}
};

struct OpKernelMapPass
    : public mlir::PassWrapper<OpKernelMapPass, mlir::OperationPass<>> {
 public:
  void runOnOperation() override {
    mlir::Operation* op = getOperation();
    op->walk([&](mlir::Operation* op) {
      LOG(INFO) << "Op.name: " << op->getName().getIdentifier().data();
      if (op->getName().getIdentifier().str() == "pd.matmul") {
        LOG(INFO) << "Process rewrite";
        pd::PDKEL_Matmul_to_CPU pattern(&getContext());
        NaivePatternRewriter rewriter(&getContext());
        if (mlir::succeeded(pattern.matchAndRewrite(op, rewriter))) {
          LOG(INFO) << "succeed in matching";
        }
        LOG(INFO) << "** after " << op->getName().getIdentifier().str();
      }
    });
  }

  llvm::StringRef getArgument() const override { return "pd-op-kernel"; }
};

void RegisterOpKernelMappingPass() {
  mlir::PassRegistration<OpKernelMapPass>();
}

std::unique_ptr<mlir::Pass> CreateOpKernelMappingPass() {
  return std::make_unique<OpKernelMapPass>();
}

}  // namespace infrt
