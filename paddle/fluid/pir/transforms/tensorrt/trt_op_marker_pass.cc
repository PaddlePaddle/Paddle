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

#include "paddle/fluid/pir/transforms/tensorrt/trt_op_marker_pass.h"
#include <glog/logging.h>
#include <bitset>
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

inline auto kCanRunTrtAttr = paddle::dialect::kCanRunTrtAttr;

#define DEFINE_GENERAL_PATTERN(OpName, OpType)                              \
  class OpName##OpPattern : public pir::OpRewritePattern<OpType> {          \
   public:                                                                  \
    using pir::OpRewritePattern<OpType>::OpRewritePattern;                  \
    bool MatchAndRewrite(OpType op,                                         \
                         pir::PatternRewriter &rewriter) const override {   \
      if (op->HasAttribute(kCanRunTrtAttr) &&                               \
          op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {       \
        VLOG(3)                                                             \
            << "Op " << #OpName                                             \
            << "already has kCanRunTrtAttr set to true. Skipping rewrite."; \
        return false;                                                       \
      }                                                                     \
      op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));          \
      return true;                                                          \
    }                                                                       \
  };

DEFINE_GENERAL_PATTERN(Matmul, paddle::dialect::MatmulOp)
DEFINE_GENERAL_PATTERN(BatchNorm, paddle::dialect::BatchNormOp)
DEFINE_GENERAL_PATTERN(BatchNorm_, paddle::dialect::BatchNorm_Op)
DEFINE_GENERAL_PATTERN(Softmax, paddle::dialect::SoftmaxOp)
DEFINE_GENERAL_PATTERN(Relu, paddle::dialect::ReluOp)
DEFINE_GENERAL_PATTERN(FullIntArray, paddle::dialect::FullIntArrayOp)
DEFINE_GENERAL_PATTERN(Reshape, paddle::dialect::ReshapeOp)
DEFINE_GENERAL_PATTERN(Dropout, paddle::dialect::DropoutOp)
DEFINE_GENERAL_PATTERN(bmm, paddle::dialect::BmmOp)
DEFINE_GENERAL_PATTERN(concat, paddle::dialect::ConcatOp)
DEFINE_GENERAL_PATTERN(flatten, paddle::dialect::FlattenOp)
#undef DEFINE_GENERAL_PATTERN

class Pool2dOpPattern
    : public pir::OpRewritePattern<paddle::dialect::Pool2dOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::Pool2dOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::Pool2dOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      VLOG(3) << "Pool2d already has kCanRunTrtAttr set to true. Skipping "
                 "rewrite.";
      return false;
    }
    auto padding_attr = op->attribute<pir::ArrayAttribute>("padding");
    std::vector<int32_t> paddings;
    for (const auto &attr : padding_attr.AsVector()) {
      paddings.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
    }
    if (paddings.size() > 2) {
      VLOG(3) << "The padding size should be less than 2";
      return false;
    }
    if (op->HasAttribute("data_format")) {
      auto data_format =
          op->attribute<pir::StrAttribute>("data_format").AsString();
      if (data_format == "NHWC" || data_format == "NDHWC") {
        VLOG(3) << "Pool2d not support NHWC or NDHWC into trt ";
        return false;
      }
    }
    if (!op->HasAttribute("pooling_type")) {
      VLOG(3) << "The pooling_type attribute does not exist";
      return false;
    } else {
      std::string pool_type =
          op->attribute<pir::StrAttribute>("pooling_type").AsString();
      if (pool_type != "max" && pool_type != "avg") {
        VLOG(3) << "Wrong pool op type, the trt do not support the "
                << pool_type << " pool type.";
        return false;
      }
      if (pool_type == "avg") {
        if (op->HasAttribute("global_pooling")) {
          if (!op->attribute<pir::BoolAttribute>("global_pooling").data()) {
            if (op->HasAttribute("exclusive")) {
              if (op->attribute<pir::BoolAttribute>("exclusive").data()) {
                paddle::dialect::FullIntArrayOp full_int_array_op =
                    pir::GetDefiningOpForInput(op, 1)
                        ->dyn_cast<paddle::dialect::FullIntArrayOp>();
                if (!full_int_array_op) {
                  VLOG(3) << "Cannot find FullIntArrayOp";
                  return false;
                } else {
                  auto attr_value =
                      full_int_array_op->attribute<pir::ArrayAttribute>(
                          "value");
                  std::vector<int32_t> kernel_size;
                  for (const auto &attr : attr_value.AsVector()) {
                    kernel_size.push_back(
                        attr.dyn_cast<pir::Int32Attribute>().data());
                  }
                  for (size_t i = 0; i < kernel_size.size(); ++i) {
                    if (kernel_size[i] <= paddings[i]) {
                      VLOG(3) << "the padding size should be less than the "
                                 "filter size "
                                 "for exclusive-counting pooling.";
                      return false;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class Conv2dOpPattern
    : public pir::OpRewritePattern<paddle::dialect::Conv2dOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::Conv2dOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::Conv2dOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      VLOG(3) << "conv2d already has kCanRunTrtAttr set to true. Skipping "
                 "rewrite.";
      return false;
    }
#if IS_TRT_VERSION_LT(7000)
    if (op->HasAttribute("padding_algorithm")) {
      std::string padding_algorithm =
          op->attribute<pir::StrAttribute>("padding_algorithm").AsString();
      if (padding_algorithm == "SAME" && op->HasAttribute("strides")) {
        auto strides_attr = op->attribute<pir::ArrayAttribute>("strides");
        std::vector<int32_t> strides;
        for (const auto &attr : strides_attr.AsVector()) {
          strides.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
        }
        if (strides.size() > 1) {
          for (size_t i = 0; i < strides.size(); ++i) {
            if (strides[i] > 1) {
              VLOG(3) << "The stride size should be 1 or less than 1";
              return false;
            }
          }
        }
      }
    }
#endif
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class Conv2dTransposeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::Conv2dTransposeOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::Conv2dTransposeOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::Conv2dTransposeOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      VLOG(3) << "conv2d_transpose already has kCanRunTrtAttr set to true. "
                 "Skipping rewrite.";
      return false;
    }
    if (!op->HasAttribute("dilations")) {
      VLOG(3) << "In conv2d_transpose, dilations attribute does not exist";
      return false;
    } else {
      auto dilation_attr = op->attribute<pir::ArrayAttribute>("dilations");
      std::vector<int32_t> dilations;
      for (const auto &attr : dilation_attr.AsVector()) {
        dilations.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
      }
      if (dilations[0] != 1 || dilations[1] != 1) {
        VLOG(3) << "In conv2d_transpose, Dilations must be (1, 1) for "
                   "tensorRT, but given ("
                << dilations[0] << ", " << dilations[1] << ")";
        return false;
      }
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class FusedConv2dAddActOpPattern
    : public pir::OpRewritePattern<paddle::dialect::FusedConv2dAddActOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::FusedConv2dAddActOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::FusedConv2dAddActOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      VLOG(3) << "fused_conv2d_add_act already has kCanRunTrtAttr set to true. "
                 "Skipping rewrite.";
      return false;
    }
#if IS_TRT_VERSION_LT(7000)
    if (op->HasAttribute("padding_algorithm")) {
      std::string padding_algorithm =
          op->attribute<pir::StrAttribute>("padding_algorithm").AsString();
      if (padding_algorithm == "SAME" && op->HasAttribute("strides")) {
        auto strides_attr = op->attribute<pir::ArrayAttribute>("strides");
        std::vector<int32_t> strides;
        for (const auto &attr : strides_attr.AsVector()) {
          strides.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
        }
        if (strides.size() > 1) {
          for (size_t i = 0; i < strides.size(); ++i) {
            if (strides[i] > 1) {
              VLOG(3) << "The stride size should be 1 or less than 1";
              return false;
            }
          }
        }
      }
    }
#endif
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class DepthwiseConv2dOpPattern
    : public pir::OpRewritePattern<paddle::dialect::DepthwiseConv2dOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::DepthwiseConv2dOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::DepthwiseConv2dOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      VLOG(3) << "depthwise_conv2d already has kCanRunTrtAttr set to true. "
                 "Skipping rewrite.";
      return false;
    }
#if IS_TRT_VERSION_LT(7000)
    if (op->HasAttribute("padding_algorithm")) {
      std::string padding_algorithm =
          op->attribute<pir::StrAttribute>("padding_algorithm").AsString();
      if (padding_algorithm == "SAME" && op->HasAttribute("strides")) {
        auto strides_attr = op->attribute<pir::ArrayAttribute>("strides");
        std::vector<int32_t> strides;
        for (const auto &attr : strides_attr.AsVector()) {
          strides.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
        }
        if (strides.size() > 1) {
          for (size_t i = 0; i < strides.size(); ++i) {
            if (strides[i] > 1) {
              VLOG(3) << "The stride size should be 1 or less than 1";
              return false;
            }
          }
        }
      }
    }
#endif
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class DepthwiseConv2dTransposeOpPattern
    : public pir::OpRewritePattern<
          paddle::dialect::DepthwiseConv2dTransposeOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::DepthwiseConv2dTransposeOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::DepthwiseConv2dTransposeOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      VLOG(3) << "depthwise_conv2d_transpose already has kCanRunTrtAttr set to "
                 "true. Skipping rewrite.";
      return false;
    }
    if (!op->HasAttribute("dilations")) {
      VLOG(3) << "In depthwise_conv2d_transpose, dilations attribute does not "
                 "exist";
      return false;
    } else {
      auto dilation_attr = op->attribute<pir::ArrayAttribute>("dilations");
      std::vector<int32_t> dilations;
      for (const auto &attr : dilation_attr.AsVector()) {
        dilations.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
      }
      if (dilations[0] != 1 || dilations[1] != 1) {
        VLOG(3)
            << "In depthwise_conv2d_transpose, Dilations must be (1, 1) for "
               "tensorRT, but given ("
            << dilations[0] << ", " << dilations[1] << ")";
        return false;
      }
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class DeformableConvOpPattern
    : public pir::OpRewritePattern<paddle::dialect::DeformableConvOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::DeformableConvOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::DeformableConvOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      VLOG(3) << "deformable_conv already has kCanRunTrtAttr set to "
                 "true. Skipping rewrite.";
      return false;
    }
    if (!op->HasAttribute("groups") || !op->HasAttribute("strides") ||
        !op->HasAttribute("paddings")) {
      VLOG(3) << "In deformable_conv, groups or strides or paddings attributes "
                 "do not exist";
      return false;
    }
    pir::Value input = op.operand_source(0);
    auto input_type = input.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto input_shape = input_type.dims();
    if (input_shape.size() != 4) {
      VLOG(3) << "Input of deformable conv should be 4-D Tensor, but got "
              << input_shape.size() << "-D Tensor";
      return false;
    }
    pir::Value filter = op.operand_source(2);
    auto filter_type =
        filter.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto filter_shape = filter_type.dims();
    int groups = op->attribute<pir::Int32Attribute>("groups").data();
    if (input_shape[1] != filter_shape[1] * groups) {
      VLOG(3) << "The number of input channels should be equal to filter "
              << "channels * groups. But got input channels " << input_shape[1]
              << "filter channels " << filter_shape[1];
      return false;
    }
    std::vector<int32_t> strides;
    auto stride_attr = op->attribute<pir::ArrayAttribute>("strides");
    for (const auto &attr : stride_attr.AsVector()) {
      strides.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
    }
    if (strides.size() != 2) {
      VLOG(3) << "The size of strides should be 2, but got " << strides.size();
      return false;
    }
    std::vector<int32_t> paddings;
    auto padding_attr = op->attribute<pir::ArrayAttribute>("paddings");
    for (const auto &attr : padding_attr.AsVector()) {
      paddings.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
    }
    if (paddings.size() != 2) {
      VLOG(3) << "The size of paddings should be 2, but got "
              << paddings.size();
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class ArangeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::ArangeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ArangeOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::ArangeOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      VLOG(3) << "arange already has kCanRunTrtAttr set to true. Skipping "
                 "rewrite.";
      return false;
    }
#if IS_TRT_VERSION_LT(8400)
    pir::Value start = op.operand_source(0);
    auto start_type = pir::GetDataTypeFromValue(start);
    if (!start_type.isa<pir::Float32Type>() ||
        !start_type.isa<pir::Float64Type>()) {
      VLOG(3) << "The type of start is not float32 or float64";
      return false;
    }
#endif
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class SignOpPattern : public pir::OpRewritePattern<paddle::dialect::SignOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SignOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::SignOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      VLOG(3) << "sign already has kCanRunTrtAttr set to true. Skipping "
                 "rewrite.";
      return false;
    }
#if IS_TRT_VERSION_LT(8200)
    VLOG(3) << "sign op is only supported by tensorrt8.2 above ";
    return false;
#endif
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class LogicalNotOpPattern
    : public pir::OpRewritePattern<paddle::dialect::LogicalNotOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::LogicalNotOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::LogicalNotOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      VLOG(3) << "logical_not already has kCanRunTrtAttr set to true. Skipping "
                 "rewrite.";
      return false;
    }
#if IS_TRT_VERSION_LT(8400)
    VLOG(3) << "logical_not op is only supported by tensorrt8.4 above because "
               "of cast op ";
    return false;
#endif
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class GroupNormOpPattern
    : public pir::OpRewritePattern<paddle::dialect::GroupNormOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::GroupNormOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::GroupNormOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      VLOG(3) << "group_norm already has kCanRunTrtAttr set to true. Skipping "
                 "rewrite.";
      return false;
    }
    if (!op->HasAttribute("epsilon") || !op->HasAttribute("groups") ||
        !op->HasAttribute("data_layout")) {
      VLOG(3) << "In group_norm, epsilon or groups or data_layout attributes "
                 "do not exist";
      return false;
    }
    std::string layout_str =
        op->attribute<pir::StrAttribute>("data_layout").AsString();
    if (layout_str != "NCHW") {
      VLOG(3) << "Group norm trt plugin only support NCHW layout, but got "
              << layout_str;
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class TransposeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::TransposeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::TransposeOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::TransposeOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      VLOG(3) << "transpose already has kCanRunTrtAttr set to true. Skipping "
                 "rewrite.";
      return false;
    }
    pir::Value x = op.operand_source(0);
    auto x_type = x.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto x_shape = x_type.dims();
    int dims = x_shape.size();
    std::vector<int> perm;
    auto perm_attr = op->attribute<pir::ArrayAttribute>("perm");
    for (const auto &attr : perm_attr.AsVector()) {
      perm.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
    }
    auto is_valid_permutation = [&](int dims,
                                    const std::vector<int> &permutation) {
      std::bitset<nvinfer1::Dims::MAX_DIMS> found;
      for (int i = 0; i < dims; ++i) {
        const int x = permutation[i];
        if ((x < 0) || (x >= dims) || found[x])
          return false;  // Out of bounds or duplicate
        found.set(x);
      }
      return true;
    };
    if (!is_valid_permutation(dims, perm)) {
      VLOG(3) << "Invalid permutation dimensions for trt transpose op "
                 "converter: duplicate or out of bound.";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class GatherOpPattern
    : public pir::OpRewritePattern<paddle::dialect::GatherOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::GatherOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::GatherOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      VLOG(3) << "gather already has kCanRunTrtAttr set to true. Skipping "
                 "rewrite.";
      return false;
    }
    pir::Value axis = op.operand_source(2);
    if (!axis) {
      VLOG(3) << "axis is empty. Skipping rewrite.";
      return false;
    }
#if IS_TRT_VERSION_LT(7000)
    pir::Value x = op.operand_source(0);
    auto x_type = x.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto x_shape = x_type.dims();
    if (x_shape.size() == 1) {
      VLOG(3) << "Gather does not support 1-dimensional input in tensorrt";
      return false;
    }
#endif
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class GatherNdOpPattern
    : public pir::OpRewritePattern<paddle::dialect::GatherNdOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::GatherNdOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::GatherNdOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      VLOG(3) << "gather_nd already has kCanRunTrtAttr set to true. Skipping "
                 "rewrite.";
      return false;
    }
#if IS_TRT_VERSION_LT(8200)
    pir::Value index_var_name = op.operand_source(1);
    auto index_var_name_type =
        index_var_name.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto index_shape = index_var_name_type.dims();
    pir::Value x_var_name = op.operand_source(0);
    auto x_var_name_type =
        x_var_name.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto x_shape = x_var_name_type.dims();
    if (x_shape.size() <= 2) {
      VLOG(3) << "gather_nd op requires the input's dimension to be greater "
                 "than 2";
      return false;
    }
    if (x_shape.size() != index_shape.size()) {
      VLOG(3) << "gather_nd op Index input dims size [" << index_shape.size()
              << " ] not equal to x dims size [" << x_shape.size() << "]";
      return false;
    }
#endif
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class TrtOpMarkerPass : public pir::PatternRewritePass {
 public:
  TrtOpMarkerPass() : pir::PatternRewritePass("trt_op_marker_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

#define ADD_PATTERN(OpName) \
  ps.Add(std::make_unique<OpName##OpPattern>(context));
    ADD_PATTERN(Matmul)
    ADD_PATTERN(BatchNorm)
    ADD_PATTERN(BatchNorm_)
    ADD_PATTERN(Softmax)
    ADD_PATTERN(Relu)
    ADD_PATTERN(FullIntArray)
    ADD_PATTERN(Reshape)
    ADD_PATTERN(Dropout)
    ADD_PATTERN(bmm)
    ADD_PATTERN(concat)
    ADD_PATTERN(flatten)
#undef ADD_PATTERN
    ps.Add(std::make_unique<Pool2dOpPattern>(context));
    ps.Add(std::make_unique<Conv2dOpPattern>(context));
    ps.Add(std::make_unique<Conv2dTransposeOpPattern>(context));
    ps.Add(std::make_unique<FusedConv2dAddActOpPattern>(context));
    ps.Add(std::make_unique<DepthwiseConv2dOpPattern>(context));
    ps.Add(std::make_unique<DepthwiseConv2dTransposeOpPattern>(context));
    ps.Add(std::make_unique<DeformableConvOpPattern>(context));
    ps.Add(std::make_unique<ArangeOpPattern>(context));
    ps.Add(std::make_unique<SignOpPattern>(context));
    ps.Add(std::make_unique<LogicalNotOpPattern>(context));
    ps.Add(std::make_unique<GroupNormOpPattern>(context));
    ps.Add(std::make_unique<TransposeOpPattern>(context));
    ps.Add(std::make_unique<GatherOpPattern>(context));
    ps.Add(std::make_unique<GatherNdOpPattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateTrtOpMarkerPass() {
  return std::make_unique<TrtOpMarkerPass>();
}
}  // namespace pir

REGISTER_IR_PASS(trt_op_marker_pass, TrtOpMarkerPass);
