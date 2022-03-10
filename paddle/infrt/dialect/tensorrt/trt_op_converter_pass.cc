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
#include <mlir/IR/Builders.h>
#include <mlir/Transforms/DialectConversion.h>
#include "paddle/infrt/dialect/infrt_base.h"
#include "paddle/infrt/dialect/pd_ops.h"
#include "paddle/infrt/dialect/tensorrt/trt_dialect_types.h"

namespace infrt {
namespace trt {

#include "paddle/infrt/dialect/tensorrt/pd_lower_to_trt.cpp.inc"  // NOLINT

struct PD2TRT_GraphLower : public ::mlir::RewritePattern {
  PD2TRT_GraphLower(::mlir::MLIRContext *context)  // NOLINT
      : ::mlir::RewritePattern("pd.graph", 1, context, {"trt.create_engine"}) {
  }  // NOLINT
  ::mlir::LogicalResult matchAndRewrite(
      ::mlir::Operation *op, ::mlir::PatternRewriter &rewriter) const override {
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
    ::mlir::Block *block = new ::mlir::Block;
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
    for (auto v : inputs) {
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

}  // namespace trt
}  // namespace infrt
