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
#include "paddle/infrt/dialect/tensorrt/trt_op_converter_pass.h"

#include <glog/logging.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "paddle/infrt/dialect/infrt/ir/infrt_dialect.h"
#include "paddle/infrt/dialect/pd/ir/pd_ops.h"
#include "paddle/infrt/dialect/tensorrt/trt_dialect_types.h"

namespace infrt {
namespace trt {

#include "paddle/infrt/dialect/tensorrt/pd_lower_to_trt.cpp.inc"  // NOLINT

struct PD2TRT_GraphLower : public ::mlir::RewritePattern {
  explicit PD2TRT_GraphLower(::mlir::MLIRContext* context)
      : ::mlir::RewritePattern("pd.graph", 1, context, {"trt.create_engine"}) {}
  ::mlir::LogicalResult matchAndRewrite(
      ::mlir::Operation* op, ::mlir::PatternRewriter& rewriter) const override {
    auto casted_op = ::llvm::dyn_cast<mlir::pd::GraphOp>(op);
    ::mlir::Operation::operand_range inputs = casted_op.inputs();
    auto ods_loc = rewriter.getFusedLoc(op->getLoc());
    CreateEngineOp create_engine_op;
    // inputs
    ::mlir::SmallVector<::mlir::Value, 4> trt_inputs;
    for (auto v : inputs) {
      trt_inputs.push_back(v);
    }
    create_engine_op = rewriter.create<CreateEngineOp>(
        ods_loc,
        ::llvm::SmallVector<mlir::Type, 4>(1, EngineType::get()),
        trt_inputs,
        true /*run_once*/);
    ::mlir::Block* block = new ::mlir::Block;

    block->getOperations().splice(block->begin(),
                                  casted_op.getBody()->getOperations(),
                                  casted_op.getBody()->begin(),
                                  casted_op.getBody()->end());
    create_engine_op.body().push_back(block);

    // trt.execute
    // outputs
    ::llvm::SmallVector<::mlir::Type, 4> execute_outputs_types;
    for (auto v : casted_op.getODSResults(0)) {
      execute_outputs_types.push_back(v.getType());
    }
    // inputs
    ::mlir::SmallVector<::mlir::Value, 4> execute_inputs(
        create_engine_op.getODSResults(0));
    for (auto v : trt_inputs) {
      execute_inputs.push_back(v);
    }
    auto execute_op = rewriter.create<ExecuteOp>(
        ods_loc, execute_outputs_types, execute_inputs);

    ::llvm::SmallVector<::mlir::Value, 4> replace_values;
    for (auto v :
         ::llvm::SmallVector<::mlir::Value, 4>{execute_op.getODSResults(0)}) {
      replace_values.push_back(v);
    }
    rewriter.replaceOp(op, replace_values);
    return ::mlir::success();
  }
};

class MyPatternRewriter : public mlir::PatternRewriter {
 public:
  explicit MyPatternRewriter(mlir::MLIRContext* ctx)
      : mlir::PatternRewriter(ctx) {}

  void replaceOp(mlir::Operation* op, mlir::ValueRange newValues) override {
    LOG(INFO) << "Replacing " << op->getName().getIdentifier().str()
              << " Inputs num:" << op->getNumOperands();

    // input converter
    for (auto v : newValues) {
      std::string str;
      llvm::raw_string_ostream os(str);
      v.print(os);
      LOG(INFO) << str;
      if (v.getDefiningOp() &&
          v.getDefiningOp()->getName().getIdentifier().str() ==
              "trt.Activation") {
        auto* def_op = v.getDefiningOp();
        ::llvm::SmallVector<::mlir::Value, 4> replace_values;
        for (size_t i = 0; i < def_op->getNumOperands(); i++) {
          RewriterBase::setInsertionPoint(def_op);
          auto cvt_tensor_op = RewriterBase::create<CvtTensorOp>(
              def_op->getLoc(),
              ::llvm::SmallVector<::mlir::Type, 4>(
                  1, def_op->getOperand(i).getType()),
              ::llvm::SmallVector<::mlir::Value, 4>(1, def_op->getOperand(i)));
          replace_values.push_back(cvt_tensor_op.getODSResults(0)[0]);
          def_op->replaceUsesOfWith(def_op->getOperand(i),
                                    cvt_tensor_op.getResult());
        }
      }
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

/*
void TRTOpConverterPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ::mlir::ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to TensorRTDialect from
  // PaddleDialect
  target.addLegalDialect<TensorRTDialect>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the TensorRT operations.
  ::mlir::RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  patterns.add<PD2TRT_GraphLower>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (::mlir::failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
*/
void TRTOpConverterPass::runOnOperation() {
  mlir::Operation* op = getOperation();

  op->walk([&](mlir::Operation* op) {
    LOG(INFO) << "walk1 Op.name: " << op->getName().getIdentifier().data();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<PD2TRT_GraphLower>(&getContext());
    populateWithGenerated(patterns);
    applyMyPatternDriver(op, std::move(patterns));
  });
}

}  // namespace trt
}  // namespace infrt
