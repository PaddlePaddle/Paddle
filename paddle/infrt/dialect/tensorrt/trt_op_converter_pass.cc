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
    ::mlir::Operation::operand_range Mean = casted_op.getODSOperands(3);
    ::mlir::Operation::operand_range Variance = casted_op.getODSOperands(4);
    operands.push_back(Input[0]);
    operands.push_back(Bias[0]);
    operands.push_back(Scale[0]);

    // TODO(weishengying) : recompute this via params
    auto *scale_producer = Scale[0].getDefiningOp();
    auto create_scale_tensor_op =
        llvm::dyn_cast<::infrt::phi::CreateHostInitedDenseTensorOp>(
            scale_producer);
    CHECK_NOTNULL(create_scale_tensor_op);

    auto *bias_producer = Bias[0].getDefiningOp();
    auto create_bias_tensor_op =
        llvm::dyn_cast<::infrt::phi::CreateHostInitedDenseTensorOp>(
            bias_producer);
    CHECK_NOTNULL(create_bias_tensor_op);

    auto *mean_producer = Mean[0].getDefiningOp();
    auto create_mean_tensor_op =
        llvm::dyn_cast<::infrt::phi::CreateHostInitedDenseTensorOp>(
            mean_producer);
    CHECK_NOTNULL(create_mean_tensor_op);

    auto *variance_producer = Variance[0].getDefiningOp();
    auto create_variance_tensor_op =
        llvm::dyn_cast<::infrt::phi::CreateHostInitedDenseTensorOp>(
            variance_producer);
    CHECK_NOTNULL(create_variance_tensor_op);

    llvm::SmallVector<double> scale_data;
    mlir::ArrayAttr scale_array_attr = create_scale_tensor_op.values();
    CHECK_GT(scale_array_attr.size(), 0U);
    CHECK(scale_array_attr[0].getType().isF32());
    scale_data.resize(scale_array_attr.size());
    for (size_t i = 0; i < scale_array_attr.size(); i++) {
      scale_data[i] =
          scale_array_attr[i].cast<mlir::FloatAttr>().getValueAsDouble();
    }

    llvm::SmallVector<double> bias_data;
    mlir::ArrayAttr bias_array_attr = create_bias_tensor_op.values();
    CHECK_GT(bias_array_attr.size(), 0U);
    CHECK(bias_array_attr[0].getType().isF32());
    bias_data.resize(bias_array_attr.size());
    for (size_t i = 0; i < bias_array_attr.size(); i++) {
      bias_data[i] =
          bias_array_attr[i].cast<mlir::FloatAttr>().getValueAsDouble();
    }

    llvm::SmallVector<double> mean_data;
    mlir::ArrayAttr mean_array_attr = create_mean_tensor_op.values();
    CHECK_GT(mean_array_attr.size(), 0U);
    CHECK(mean_array_attr[0].getType().isF32());
    mean_data.resize(mean_array_attr.size());
    for (size_t i = 0; i < mean_array_attr.size(); i++) {
      mean_data[i] =
          mean_array_attr[i].cast<mlir::FloatAttr>().getValueAsDouble();
    }

    llvm::SmallVector<double> variance_data;
    mlir::ArrayAttr variance_array_attr = create_variance_tensor_op.values();
    CHECK_GT(variance_array_attr.size(), 0U);
    CHECK(variance_array_attr[0].getType().isF32());
    variance_data.resize(variance_array_attr.size());
    for (size_t i = 0; i < variance_array_attr.size(); i++) {
      variance_data[i] =
          variance_array_attr[i].cast<mlir::FloatAttr>().getValueAsDouble();
    }

    double eps = casted_op.epsilonAttr().getValueAsDouble();

    llvm::SmallVector<float> combile_scale_data;
    combile_scale_data.resize(scale_data.size());
    llvm::SmallVector<float> combile_bias_data;
    combile_bias_data.resize(bias_data.size());

    size_t ele_num = combile_scale_data.size();
    for (size_t i = 0; i < ele_num; i++) {
      float scale = scale_data[i];
      float bias = bias_data[i];
      float mean = mean_data[i];
      float variance = variance_data[i];
      combile_scale_data[i] = scale / sqrtf(variance + eps);
      combile_bias_data[i] = bias - mean * combile_scale_data[i];
    }

    rewriter.setInsertionPoint(create_scale_tensor_op);
    auto new_scale_op =
        rewriter.create<::infrt::phi::CreateHostInitedDenseTensorOp>(
            create_scale_tensor_op->getLoc(),
            create_scale_tensor_op.output().getType(),
            create_scale_tensor_op.context(),
            create_scale_tensor_op.dims(),
            ::infrt::LayoutAttr::get(rewriter.getContext(),
                                     ::infrt::LayoutType::NCHW),
            create_scale_tensor_op.lod(),
            rewriter.getF32ArrayAttr(combile_scale_data));
    rewriter.replaceOp(create_scale_tensor_op, new_scale_op->getResults());

    rewriter.setInsertionPoint(create_bias_tensor_op);
    auto new_bias_op =
        rewriter.create<::infrt::phi::CreateHostInitedDenseTensorOp>(
            create_bias_tensor_op->getLoc(),
            create_bias_tensor_op.output().getType(),
            create_bias_tensor_op.context(),
            create_bias_tensor_op.dims(),
            ::infrt::LayoutAttr::get(rewriter.getContext(),
                                     ::infrt::LayoutType::NCHW),
            create_bias_tensor_op.lod(),
            rewriter.getF32ArrayAttr(combile_bias_data));
    rewriter.replaceOp(create_bias_tensor_op, new_bias_op->getResults());

    trt::ScaleNdOp scaleNd_op;
    // resultTypes
    ::mlir::SmallVector<::mlir::Type, 4> resultTypes;
    for (auto v : casted_op.getODSResults(0)) {
      resultTypes.push_back(v.getType());
    }

    // attributes
    rewriter.setInsertionPoint(op);
    ::mlir::SmallVector<::mlir::NamedAttribute, 8> attributes;
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

std::unique_ptr<mlir::Pass> CreateTrtOpConverterPass() {
  return std::make_unique<TRTOpConverterPass>();
}

}  // namespace trt
}  // namespace infrt
