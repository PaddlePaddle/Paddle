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
#include <mlir/Transforms/DialectConversion.h>

#include "paddle/infrt/dialect/dense_tensor.h"
#include "paddle/infrt/dialect/pd/ir/pd_ops.h"
#include "paddle/infrt/dialect/phi/ir/infrt_phi_tensor.h"
#include "paddle/infrt/dialect/phi/ir/phi_base.h"
#include "paddle/infrt/dialect/tensorrt/convert.h"
#include "paddle/infrt/dialect/tensorrt/trt_dialect_types.h"
#include "paddle/infrt/dialect/tensorrt/trt_ops.h"

namespace infrt {
namespace trt {

#ifdef INFRT_WITH_TRT

#define STRING_TO_ENUM_TYPE(enum_type) enum_type
#define STRING_TO_ENUM_VALUE(enum_value) enum_value
#include <NvInfer.h>

#else  // INFRT_WITH_TRT

#define STRING_TO_ENUM_TYPE(enum_type) std::string
#define STRING_TO_ENUM_VALUE(enum_value) #enum_value

#endif  // INFRT_WITH_TRT

template <typename T>
::mlir::IntegerAttr createNvinferEnumAttr(
    ::mlir::PatternRewriter &rewriter,  // NOLINT
    T enum_value) {
  return rewriter.getSI32IntegerAttr((int32_t)enum_value);
}

template <>
::mlir::IntegerAttr createNvinferEnumAttr<std::string>(
    ::mlir::PatternRewriter &rewriter, std::string enum_value) {  // NOLINT
  (void)enum_value;
  return rewriter.getSI32IntegerAttr(-1);
}

#include "paddle/infrt/dialect/tensorrt/pd_lower_to_trt.cpp.inc"  // NOLINT

struct PD2TRT_GraphLower : public ::mlir::RewritePattern {
  explicit PD2TRT_GraphLower(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern(
            "infrt.graph", 1, context, {"trt.create_engine"}) {}
  ::mlir::LogicalResult matchAndRewrite(
      ::mlir::Operation *op, ::mlir::PatternRewriter &rewriter) const override {
    auto casted_op = ::llvm::dyn_cast<::infrt::GraphOp>(op);
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
    auto &block = create_engine_op.body().emplaceBlock();
    block.getOperations().splice(block.begin(),
                                 casted_op.getBody()->getOperations(),
                                 casted_op.getBody()->begin(),
                                 casted_op.getBody()->end());

    // trt.compute
    ::llvm::SmallVector<::mlir::Value, 4> replace_values2;
    auto ctx_op = rewriter.create<::infrt::phi::CreateGPUContextOp>(
        ods_loc,
        infrt::phi::ContextType::get(rewriter.getContext(),
                                     infrt::TargetType::GPU));
    auto compute_op = rewriter.create<EngineComputeOp>(
        ods_loc,
        ::infrt::DenseTensorListType::get(rewriter.getContext()),
        create_engine_op.engine(),
        ctx_op.output());
    auto tensor_list_val = compute_op.outputs();
    for (size_t i = 0; i < casted_op.getNumResults(); ++i) {
      auto res = casted_op->getResult(i);
      auto int_attr = mlir::IntegerAttr::get(
          mlir::IntegerType::get(rewriter.getContext(), 32), i);
      auto get_tensor_op = rewriter.create<::infrt::dt::TensorListGetTensorOp>(
          ods_loc, res.getType(), tensor_list_val, int_attr);
      replace_values2.push_back(get_tensor_op.output());
    }
    ctx_op->moveBefore(ctx_op->getBlock(), ctx_op->getBlock()->begin());
    rewriter.replaceOp(op, replace_values2);
    return ::mlir::success();
  }
};

struct PD2TRT_Batch_Norm_Lower : public ::mlir::RewritePattern {
  explicit PD2TRT_Batch_Norm_Lower(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("pd.batch_norm", 1, context, {"trt.scaleNd"}) {}
  ::mlir::LogicalResult matchAndRewrite(
      ::mlir::Operation *op, ::mlir::PatternRewriter &rewriter) const override {
    auto casted_op = ::llvm::dyn_cast<infrt::pd::Batch_normOp>(op);
    ::mlir::SmallVector<::mlir::Value, 4> operands;
    ::mlir::Operation::operand_range Input = casted_op.getODSOperands(0);
    ::mlir::Operation::operand_range Scale = casted_op.getODSOperands(1);
    ::mlir::Operation::operand_range Bias = casted_op.getODSOperands(2);

    // TODO(weishengying) : recompute this via params
    operands.push_back((*Input.begin()));
    operands.push_back((*Scale.begin()));
    operands.push_back((*Bias.begin()));
    operands.push_back((*Bias.begin()));

    trt::ScaleNdOp scaleNd_op;
    // inputs
    ::mlir::SmallVector<::mlir::Value, 4> trt_inputs;
    for (auto v : operands) {
      trt_inputs.push_back(v);
    }

    // resultTypes
    ::mlir::SmallVector<::mlir::Type, 4> resultTypes;
    for (auto v : casted_op.getODSResults(0)) {
      resultTypes.push_back(v.getType());
    }

    // attributes
    ::mlir::SmallVector<::mlir::NamedAttribute, 8> attributes;
    {
      auto mode_attr = rewriter.getI32IntegerAttr(1);
      attributes.emplace_back(rewriter.getStringAttr("mode"), mode_attr);
    }

    {
      auto axis_attr = rewriter.getI32IntegerAttr(-1);
      attributes.emplace_back(rewriter.getStringAttr("axis"), axis_attr);
    }
    auto result = rewriter
                      .create<trt::ScaleNdOp>(
                          op->getLoc(), resultTypes, operands, attributes)
                      .getODSResults(0);
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    // TODO(weishengying) : update it
    for (uint32_t i = 0; i < casted_op.getNumResults(); i++) {
      for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{result}) {
        tblgen_repl_values.push_back(v);
      }
    }
    rewriter.replaceOp(op, tblgen_repl_values);
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
  target.addLegalDialect<::infrt::phi::PHIDialect>();
  target.addLegalDialect<::infrt::dt::DTDialect>();
  target.addLegalDialect<phi::PHIDenseTensorDialect>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the TensorRT operations.
  ::mlir::RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  patterns.add<PD2TRT_Batch_Norm_Lower>(&getContext());
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
