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
#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "paddle/infrt/dialect/infrt_base.h"
#include "paddle/infrt/dialect/pd_ops.h"

namespace infrt {
namespace pd {
#include "paddle/infrt/dialect/rewrite.hpp.inc"
}  // namespace pd
}  // namespace infrt

namespace infrt {

class MyPatternRewriter : public mlir::PatternRewriter {
 public:
  explicit MyPatternRewriter(mlir::MLIRContext* ctx)
      : mlir::PatternRewriter(ctx) {}

  void replaceOp(mlir::Operation* op, mlir::ValueRange newValues) override {
    LOG(INFO) << "Replacing " << op->getName().getIdentifier().str();
    for (size_t i = 0; i < newValues.size(); i++) {
      std::string str;
      llvm::raw_string_ostream os(str);
      newValues[i].print(os);
      LOG(INFO) << str;
    }
    RewriterBase::replaceOp(op, newValues);
  }
};

/// Apply the custom driver to `op`.
void applyMyPatternDriver(mlir::Operation* op,
                          mlir::RewritePatternSet&& patterns) {
  // Initialize the custom PatternRewriter.
  MyPatternRewriter rewriter(op->getContext());

  mlir::FrozenRewritePatternSet fpatterns(std::move(patterns));
  // Create the applicator and apply our cost model.
  mlir::PatternApplicator applicator(fpatterns);
  applicator.applyCostModel([](const mlir::Pattern& pattern) {
    // Apply a default cost model.
    // Note: This is just for demonstration, if the default cost model is truly
    //       desired `applicator.applyDefaultCostModel()` should be used
    //       instead.
    return pattern.getBenefit();
  });

  // Try to match and apply a pattern.
  auto result = applicator.matchAndRewrite(op, rewriter);
  if (failed(result)) {
    LOG(INFO) << "No pattern applied";
  }
  // ... A pattern was successfully applied.
}

struct OpKernelMapPass
    : public mlir::PassWrapper<OpKernelMapPass, mlir::OperationPass<>> {
 public:
  void runOnOperation() override {
    mlir::Operation* op = getOperation();

    op->walk([&](mlir::Operation* op) {
      LOG(INFO) << "Op.name: " << op->getName().getIdentifier().data();
      if (op->getName().getIdentifier().str() == "pd.matmul") {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<pd::PDKEL_Matmul_to_CPU>(patterns.getContext());

        applyMyPatternDriver(op, std::move(patterns));
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
