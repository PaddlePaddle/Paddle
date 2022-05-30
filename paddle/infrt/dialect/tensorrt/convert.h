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
#pragma once

#include <glog/logging.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include "paddle/infrt/dialect/infrt/common/types.h"
#include "paddle/infrt/dialect/infrt/ir/infrt_dialect.h"
#include "paddle/infrt/dialect/pd/ir/pd_ops.h"
#include "paddle/infrt/dialect/phi/ir/infrt_phi_tensor.h"
#include "paddle/infrt/dialect/tensorrt/trt_ops.h"
#include "paddle/infrt/kernel/tensorrt/trt_helper.h"

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

static mlir::Value createTRTConv2dOp(mlir::PatternRewriter &rewriter,  // NOLINT
                                     mlir::Operation *op,
                                     mlir::Value input,
                                     mlir::Value filter) {
  auto conv_op = ::llvm::dyn_cast<infrt::pd::Conv2dOp>(op);
  ::mlir::SmallVector<::mlir::Value, 4> operands;
  operands.push_back(input);
  operands.push_back(filter);

  ::mlir::SmallVector<::mlir::Type, 4> resultTypes;
  for (auto v : conv_op.getODSResults(0)) {
    resultTypes.push_back(v.getType());
  }

  ::mlir::SmallVector<::mlir::NamedAttribute, 8> attributes;

  auto *filter_producer = filter.getDefiningOp();
  auto create_inited_tensor_op =
      llvm::dyn_cast<::infrt::phi::CreateHostInitedDenseTensorOp>(
          filter_producer);

  CHECK_NOTNULL(create_inited_tensor_op);
  mlir::ArrayAttr dims = create_inited_tensor_op.dims();
  CHECK_EQ(dims.size(), 4U);
  CHECK(dims[0].getType().isIntOrIndex());

  const int32_t n_output = dims[0].cast<mlir::IntegerAttr>().getInt();
  const int32_t filter_h = dims[2].cast<mlir::IntegerAttr>().getInt();
  const int32_t filter_w = dims[3].cast<mlir::IntegerAttr>().getInt();

  auto padding_attr = conv_op->getAttrOfType<::mlir::ArrayAttr>("paddings");
  llvm::SmallVector<int32_t, 4> paddings(padding_attr.size());
  for (size_t i = 0; i < padding_attr.size(); i++) {
    paddings[i] = padding_attr[i].cast<mlir::IntegerAttr>().getInt();
  }

  auto dilations_attr = conv_op->getAttrOfType<::mlir::ArrayAttr>("dilations");
  llvm::SmallVector<int32_t> dilations(dilations_attr.size());
  for (size_t i = 0; i < dilations_attr.size(); i++) {
    dilations[i] = dilations_attr[i].cast<mlir::IntegerAttr>().getInt();
  }

  llvm::SmallVector<int32_t, 2> nv_paddings(2);
  llvm::SmallVector<int32_t, 4> nv_pre_paddings(2);
  llvm::SmallVector<int32_t, 4> nv_post_paddings(2);
  llvm::SmallVector<int32_t, 2> nv_dilations({dilations[0], dilations[1]});
  int32_t nv_padding_mode = 0;  // nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN
  auto padding_algorithm_attr =
      conv_op->getAttrOfType<::mlir::StringAttr>("padding_algorithm");
  if (padding_algorithm_attr.strref() == "VALID") {
    for (size_t i = 0; i < paddings.size(); i++) {
      paddings[i] = 0;
    }
  }
  if (padding_algorithm_attr.strref() == "SAME") {
    nv_padding_mode = 2;  // nvinfer1::PaddingMode::kSAME_UPPER
    nv_dilations[0] = 1;
    nv_dilations[1] = 1;
  }

  if (paddings.size() == 2) {
    nv_paddings[0] = paddings[0];
    nv_paddings[1] = paddings[1];
  } else {
    CHECK_EQ(paddings.size(), 4U);
    nv_pre_paddings[0] = paddings[0];
    nv_pre_paddings[1] = paddings[2];
    nv_post_paddings[0] = paddings[1];
    nv_post_paddings[1] = paddings[3];
  }

  attributes.emplace_back(rewriter.getStringAttr("out_channel_num"),
                          rewriter.getSI32IntegerAttr(n_output));

  attributes.emplace_back(rewriter.getStringAttr("kernel_size"),
                          rewriter.getI32ArrayAttr({filter_h, filter_w}));

  attributes.emplace_back(
      rewriter.getStringAttr("dilations"),
      rewriter.getI32ArrayAttr({nv_dilations[0], nv_dilations[1]}));

  attributes.emplace_back(rewriter.getStringAttr("padding_mode"),
                          rewriter.getSI32IntegerAttr(nv_padding_mode));

  attributes.emplace_back(rewriter.getStringAttr("paddings"),
                          rewriter.getI32ArrayAttr({paddings[0], paddings[1]}));

  attributes.emplace_back(
      rewriter.getStringAttr("pre_paddings"),
      rewriter.getI32ArrayAttr({nv_pre_paddings[0], nv_pre_paddings[1]}));

  attributes.emplace_back(
      rewriter.getStringAttr("post_paddings"),
      rewriter.getI32ArrayAttr({nv_post_paddings[0], nv_post_paddings[1]}));

  {
    auto tblgen_attr = conv_op->getAttrOfType<::mlir::IntegerAttr>("groups");
    attributes.emplace_back(rewriter.getStringAttr("groups"), tblgen_attr);
  }
  {
    auto tblgen_attr = conv_op->getAttrOfType<::mlir::ArrayAttr>("strides");
    attributes.emplace_back(rewriter.getStringAttr("strides"), tblgen_attr);
  }
  return rewriter.create<trt::ConvolutionOp>(
      conv_op->getLoc(), resultTypes, operands, attributes);
}

static inline mlir::ArrayAttr TransposeWeight(
    mlir::PatternRewriter &builder,  // NOLINT
    const mlir::ArrayAttr &weight,
    const mlir::ArrayAttr &dims) {
  CHECK_EQ(dims.size(), 2U);
  CHECK(!dims.empty());
  CHECK(dims[0].getType().isInteger(64));
  CHECK(!weight.empty());
  CHECK(weight[0].getType().isF32());

  int row = dims[0].cast<mlir::IntegerAttr>().getInt();
  int col = dims[1].cast<mlir::IntegerAttr>().getInt();
  std::vector<float> trans_weight(weight.size());
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
      trans_weight[j * row + i] =
          weight[i * col + j].cast<mlir::FloatAttr>().getValueAsDouble();
    }
  }
  return builder.getF32ArrayAttr(trans_weight);
}

// matmul_y and elt_y is weights.
inline ::llvm::SmallVector<::mlir::Value, 4> createTrtFcOp(
    mlir::PatternRewriter &builder,  // NOLINT
    mlir::Value matmul_x,
    mlir::Value matmul_y,
    mlir::Value elt_y,
    mlir::Value elt_out) {
  ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

  auto *y_producer = matmul_y.getDefiningOp();
  auto create_inited_tensor_op =
      llvm::dyn_cast<::infrt::phi::CreateHostInitedDenseTensorOp>(y_producer);
  CHECK_NOTNULL(create_inited_tensor_op);

  mlir::ArrayAttr dims = create_inited_tensor_op.dims();
  CHECK_EQ(dims.size(), 2U);

  std::vector<int64_t> new_dims(dims.size());
  CHECK(!dims.empty());
  CHECK(dims[0].getType().isIntOrIndex());
  for (size_t i = 0; i < new_dims.size(); ++i) {
    new_dims[i] = dims[dims.size() - 1 - i].cast<mlir::IntegerAttr>().getInt();
  }
  auto insert_point = builder.saveInsertionPoint();
  builder.setInsertionPoint(create_inited_tensor_op);
  auto new_inited_op =
      builder.create<::infrt::phi::CreateHostInitedDenseTensorOp>(
          create_inited_tensor_op->getLoc(),
          create_inited_tensor_op.output().getType(),
          create_inited_tensor_op.context(),
          builder.getI64ArrayAttr(new_dims),
          ::infrt::LayoutAttr::get(builder.getContext(),
                                   ::infrt::LayoutType::NCHW),
          create_inited_tensor_op.lod(),
          TransposeWeight(builder, create_inited_tensor_op.values(), dims));
  builder.replaceOp(create_inited_tensor_op, new_inited_op->getResults());
  builder.restoreInsertionPoint(insert_point);

  auto ods_loc = builder.getFusedLoc({y_producer->getLoc()});
  ::infrt::trt::FullyConnectedOp fc_op;
  {
    ::mlir::SmallVector<::mlir::Type, 4> tblgen_types;

    fc_op = builder.create<::infrt::trt::FullyConnectedOp>(
        ods_loc,
        elt_out.getType(),
        matmul_x,
        new_inited_op.output(),
        elt_y,
        builder.getSI32IntegerAttr(new_dims[0]));
  }

  ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
  for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{fc_op.getODSResults(0)}) {
    tblgen_repl_values.push_back(v);
  }
  return tblgen_repl_values;
}

inline mlir::IntegerAttr CreatePoolingType(
    mlir::PatternRewriter &builder,  // NOLINT
    mlir::StringAttr pool_type) {
  // pool_type.
  auto ptype = pool_type.str();
  if (ptype == "max") {
    return createNvinferEnumAttr(builder, nvinfer1::PoolingType::kMAX);
  } else if (ptype == "avg") {
    return createNvinferEnumAttr(builder, nvinfer1::PoolingType::kAVERAGE);
  } else {
    llvm_unreachable("unknown pool_type.");
    return {};
  }
}

inline mlir::IntegerAttr CreatePaddingMode(
    mlir::PatternRewriter &builder,  // NOLINT
    mlir::StringAttr padding_algorithm,
    mlir::BoolAttr ceil_mode) {
  // TODO(Inference): Phi pool kernel seems not process ceil_mode.
  auto padding_algo = padding_algorithm.str();
  if (padding_algo == "SAME") {
    return createNvinferEnumAttr(builder, nvinfer1::PaddingMode::kSAME_UPPER);
  }
  if (ceil_mode.getValue() && padding_algo != "SAME") {
    return createNvinferEnumAttr(builder,
                                 nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP);
  } else {
    return createNvinferEnumAttr(builder,
                                 nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN);
  }
}

inline ::llvm::SmallVector<::mlir::Value, 4> CreatePaddleTrtPoolingOp(
    mlir::PatternRewriter &builder,  // NOLINT
    mlir::Value input,
    mlir::StringAttr pool_type,
    mlir::ArrayAttr ksize,
    mlir::BoolAttr global_pooling,
    mlir::ArrayAttr strides,
    mlir::ArrayAttr paddings,
    mlir::BoolAttr exclusive,
    mlir::BoolAttr adaptive,
    mlir::BoolAttr ceil_mode,
    mlir::StringAttr data_format,
    mlir::StringAttr padding_algorithm) {
  ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;

  // TODO(inference): Support NHWC.
  if (data_format.str() != "NCHW") {
    CHECK(false) << "The pool2d converter now only support NCHW.";
  }

  // TODO(Wilber): How to support dynamic shape?

  auto *input_producer = input.getDefiningOp();

  // Process pool_type.
  auto pool_type_attr = CreatePoolingType(builder, pool_type);

  // Update padding.
  auto padding_algorithm_str = padding_algorithm.str();
  auto paddings_attr = paddings;
  if (padding_algorithm_str == "EXPLICIT") {
    // Do nothing on paddings.
  } else if (padding_algorithm_str == "SAME") {
    // We should process this case in trt network build phase.
  } else if (padding_algorithm_str == "VALID") {
    // Set padding to zero.
    paddings_attr = builder.getI32ArrayAttr({0, 0});
  } else {
    CHECK(false) << "Unknown padding_algotithm.";
  }

  // if global_pooling == true or adaptive == true, padding will be ignored
  // if (global_pooling.getValue() || adaptive.getValue()) {
  //   paddings_attr = builder.getI32ArrayAttr({0, 0});
  // }

  // if global_pooling == true, then we should update kernel size to input dims.
  if (global_pooling.getValue() == true) {
    // Update ksize to input dims.
  }

  // The adaptive logic should be processed when we get the context of
  // INetworkDefinition, so we place the logic in infrt runtime(trt compile
  // time).

  // The `exclusive` may be a naive attr, which can be forward to trt.

  auto padding_mode_attr =
      CreatePaddingMode(builder, padding_algorithm, ceil_mode);

  if (global_pooling.getValue() == true) {
    CHECK(false) << "Temporarily not support global_pool";
    return tblgen_repl_values;
  }

  PoolingOp pool_op;
  {
    auto ods_loc = builder.getFusedLoc({input_producer->getLoc()});
    pool_op = builder.create<PoolingOp>(ods_loc,
                                        input.getType(),
                                        input,
                                        pool_type_attr,
                                        ksize,
                                        strides,
                                        paddings_attr,
                                        padding_mode_attr,
                                        exclusive,
                                        adaptive,
                                        padding_algorithm);
  }

  for (auto v :
       ::llvm::SmallVector<::mlir::Value, 4>{pool_op.getODSResults(0)}) {
    tblgen_repl_values.push_back(v);
  }
  return tblgen_repl_values;
}

}  // namespace trt
}  // namespace infrt
