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
#include <vector>
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

#define DEFINE_GENERAL_PATTERN(OpName, OpType)                            \
  class OpName##OpPattern : public pir::OpRewritePattern<OpType> {        \
   public:                                                                \
    using pir::OpRewritePattern<OpType>::OpRewritePattern;                \
    bool MatchAndRewrite(OpType op,                                       \
                         pir::PatternRewriter &rewriter) const override { \
      if (op->HasAttribute(kCanRunTrtAttr) &&                             \
          op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {     \
        return false;                                                     \
      }                                                                   \
      op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));        \
      return true;                                                        \
    }                                                                     \
  };

DEFINE_GENERAL_PATTERN(Matmul, paddle::dialect::MatmulOp)
DEFINE_GENERAL_PATTERN(BatchNorm, paddle::dialect::BatchNormOp)
DEFINE_GENERAL_PATTERN(BatchNorm_, paddle::dialect::BatchNorm_Op)
DEFINE_GENERAL_PATTERN(Softmax, paddle::dialect::SoftmaxOp)
DEFINE_GENERAL_PATTERN(Relu, paddle::dialect::ReluOp)
DEFINE_GENERAL_PATTERN(FullIntArray, paddle::dialect::FullIntArrayOp)
DEFINE_GENERAL_PATTERN(Reshape, paddle::dialect::ReshapeOp)
DEFINE_GENERAL_PATTERN(Dropout, paddle::dialect::DropoutOp)
DEFINE_GENERAL_PATTERN(Bmm, paddle::dialect::BmmOp)
DEFINE_GENERAL_PATTERN(Concat, paddle::dialect::ConcatOp)
DEFINE_GENERAL_PATTERN(Nonzero, paddle::dialect::NonzeroOp)
DEFINE_GENERAL_PATTERN(Gelu, paddle::dialect::GeluOp)
DEFINE_GENERAL_PATTERN(Fused_gemm_epilogue,
                       paddle::dialect::FusedGemmEpilogueOp)
DEFINE_GENERAL_PATTERN(Layer_norm, paddle::dialect::LayerNormOp)
DEFINE_GENERAL_PATTERN(Add, paddle::dialect::AddOp)
DEFINE_GENERAL_PATTERN(Full, paddle::dialect::FullOp)
DEFINE_GENERAL_PATTERN(Silu, paddle::dialect::SiluOp)
DEFINE_GENERAL_PATTERN(Conv2d, paddle::dialect::Conv2dOp)
DEFINE_GENERAL_PATTERN(FusedConv2dAddAct, paddle::dialect::FusedConv2dAddActOp)
DEFINE_GENERAL_PATTERN(DepthwiseConv2d, paddle::dialect::DepthwiseConv2dOp)
DEFINE_GENERAL_PATTERN(Sigmoid, paddle::dialect::SigmoidOp)

#undef DEFINE_GENERAL_PATTERN

class Pool2dOpPattern
    : public pir::OpRewritePattern<paddle::dialect::Pool2dOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::Pool2dOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::Pool2dOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    auto padding_attr = op->attribute<pir::ArrayAttribute>("paddings");
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
                  std::vector<int64_t> kernel_size;
                  for (const auto &attr : attr_value.AsVector()) {
                    kernel_size.push_back(
                        attr.dyn_cast<pir::Int64Attribute>().data());
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

class Conv2dTransposeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::Conv2dTransposeOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::Conv2dTransposeOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::Conv2dTransposeOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
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
      return false;
    }
    if (!op->HasAttribute("epsilon") || !op->HasAttribute("groups") ||
        !op->HasAttribute("data_format")) {
      VLOG(3) << "In group_norm, epsilon or groups or data_format attributes "
                 "do not exist";
      return false;
    }
    std::string layout_str =
        op->attribute<pir::StrAttribute>("data_format").AsString();
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
      return false;
    }
    pir::Value axis = op.operand_source(2);
    if (!axis) {
      VLOG(3) << "axis is empty. Skipping rewrite.";
      return false;
    }
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

class ScaleOpPattern : public pir::OpRewritePattern<paddle::dialect::ScaleOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ScaleOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::ScaleOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    pir::Value x = op.operand_source(0);
    auto x_dtype = pir::GetDataTypeFromValue(x);
    if (!(x_dtype.isa<pir::Float32Type>() || x_dtype.isa<pir::Float64Type>() ||
          x_dtype.isa<pir::Float16Type>() || x_dtype.isa<pir::Int32Type>() ||
          x_dtype.isa<pir::Int64Type>())) {
      VLOG(3) << "At present, ScaleOp only support float32 or float16 or "
                 "float64 or int32 or int64 into trt.";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class UnsqueezeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::UnsqueezeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::UnsqueezeOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::UnsqueezeOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    paddle::dialect::FullIntArrayOp full_int_array_op =
        pir::GetDefiningOpForInput(op, 1)
            ->dyn_cast<paddle::dialect::FullIntArrayOp>();
    auto axis = full_int_array_op->attribute<pir::ArrayAttribute>("value");

    if (!axis) {
      VLOG(3) << "The necessary attributes of the unsuqeeze axis is missing";
      return false;
    }
    pir::Value x = op.operand_source(0);
    auto x_type = x.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto x_shape = x_type.dims();

    std::vector<int32_t> dynamic_dims;
    for (int i = 0; i < x_shape.size(); ++i) {
      if (x_shape[i] == -1) {
        dynamic_dims.push_back(i);
      }
    }
    if (dynamic_dims.size() > 1) {
      VLOG(3) << "Currently we don't support unsqueeze with more than one "
                 "dynamic dims";
      return false;
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class Unsqueeze_OpPattern
    : public pir::OpRewritePattern<paddle::dialect::Unsqueeze_Op> {
 public:
  using pir::OpRewritePattern<paddle::dialect::Unsqueeze_Op>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::Unsqueeze_Op op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    paddle::dialect::FullIntArrayOp full_int_array_op =
        pir::GetDefiningOpForInput(op, 1)
            ->dyn_cast<paddle::dialect::FullIntArrayOp>();
    auto axis = full_int_array_op->attribute<pir::ArrayAttribute>("value");

    if (!axis) {
      VLOG(3) << "The necessary attributes of the unsuqeeze axis is missing";
      return false;
    }
    pir::Value x = op.operand_source(0);
    auto x_type = x.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto x_shape = x_type.dims();

    std::vector<int32_t> dynamic_dims;
    for (int i = 0; i < x_shape.size(); ++i) {
      if (x_shape[i] == -1) {
        dynamic_dims.push_back(i);
      }
    }
    if (dynamic_dims.size() > 1) {
      VLOG(3) << "Currently we don't support unsqueeze with more than one "
                 "dynamic dims";
      return false;
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class SqueezeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::SqueezeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SqueezeOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::SqueezeOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }

    paddle::dialect::FullIntArrayOp full_int_array_op =
        pir::GetDefiningOpForInput(op, 1)
            ->dyn_cast<paddle::dialect::FullIntArrayOp>();

    auto axis = full_int_array_op->attribute<pir::ArrayAttribute>("value");
    std::vector<int64_t> axes;
    for (const auto &attr : axis.AsVector()) {
      axes.push_back(attr.dyn_cast<pir::Int64Attribute>().data());
    }
    if (axes.empty()) {
      auto input_var_name = op.operand_source(0);
      auto input_var_name_type =
          input_var_name.type().dyn_cast<paddle::dialect::DenseTensorType>();
      auto input_var_name_shape = input_var_name_type.dims();

      for (int i = 0; i < input_var_name_shape.size(); ++i) {
        int64_t s = input_var_name_shape[i];
        if (s == -1) {
          VLOG(3) << "The necessary attributes of the squeeze operator axis is "
                     "missing. ss =====-1";
          return false;
        } else if (s == 1) {
          axes.push_back(s);
        }
      }

      if (axes.empty()) {
        VLOG(3) << "The necessary attributes of the squeeze2 operator axes is "
                   "missing.";
        return false;
      }
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class SliceOpPattern : public pir::OpRewritePattern<paddle::dialect::SliceOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SliceOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::SliceOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }

    if (!op->HasAttribute("axes")) {
      VLOG(3)
          << "The necessary attribute of the slice operator axes are missing.";
      return false;
    }

    auto axes_attr = op->attribute<pir::ArrayAttribute>("axes");

    std::vector<int64_t> axes;
    for (const auto &attr : axes_attr.AsVector()) {
      axes.push_back(attr.dyn_cast<pir::Int64Attribute>().data());
    }
    pir::Value input = op.operand_source(0);

    auto inputs = input.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto inputs_shape = inputs.dims();
    if (axes.size() !=
        static_cast<std::vector<int64_t>::size_type>(inputs_shape.size())) {
      VLOG(3) << "The shape of attributes of the slice operator axes "
                 "and starts are not equal.";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class IndexSelectOpPattern
    : public pir::OpRewritePattern<paddle::dialect::IndexSelectOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::IndexSelectOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::IndexSelectOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
#if IS_TRT_VERSION_LT(8200)
    VLOG(3) << "index_select op is only supported by tensorrt8.2 above ";
    return false;
#endif
    pir::Value x = op.operand_source(1);
    auto x_dtype = pir::GetDataTypeFromValue(x);
    if (!(x_dtype.isa<pir::Int32Type>() || x_dtype.isa<pir::Int64Type>())) {
      VLOG(3) << "Index select op Index input data type must be int32 or int64";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class FlattenOpPattern
    : public pir::OpRewritePattern<paddle::dialect::FlattenOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::FlattenOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::FlattenOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    if (!op->HasAttribute("start_axis") && !op->HasAttribute("stop_axis")) {
      VLOG(3) << "flatten op must has start_axis and stop_axis attributes";
      return false;
    }
    int start_axis = op->attribute<pir::Int32Attribute>("start_axis").data();
    int stop_axis = op->attribute<pir::Int32Attribute>("stop_axis").data();

    pir::Value x = op.operand_source(0);
    auto x_type = x.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto x_shape = x_type.dims();
    int dims = x_shape.size();
    if (dims == 0) {
      VLOG(3) << "Flatten op does not support input's dim is 0 in tensorrt "
                 "static shape mode.";
    }
    if (start_axis < 0) {
      start_axis += dims;
    }

    if (start_axis == 0) {
      VLOG(3) << "TRT flatten_contiguous_range not support the "
                 "batch-dimension being changed";
      return false;
    }
    if (stop_axis < 0) {
      stop_axis += dims;
    }
    for (int i = start_axis; i <= stop_axis; ++i) {
      if (x_shape[i] < 0) {
        VLOG(3) << "On TRT static shape,flatten_contiguous_range input dim "
                   "should be > 0";
        return false;
      }
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};
class CastOpPattern : public pir::OpRewritePattern<paddle::dialect::CastOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::CastOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::CastOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    if (!op->HasAttribute("dtype")) {
      VLOG(3) << "the cast op does not have attr dtype ";
      return false;
    }
    auto dtype =
        op->attribute<paddle::dialect::DataTypeAttribute>("dtype").data();
    if (dtype == phi::DataType::BOOL) {
#if IS_TRT_VERSION_LT(8400)
      VLOG(3)
          << "the cast op supports inputs and outputs of BOOL by trt8.4 above ";
      return false;
#endif
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class SplitOpPattern : public pir::OpRewritePattern<paddle::dialect::SplitOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SplitOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::SplitOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }

    paddle::dialect::FullOp full_op =
        pir::GetDefiningOpForInput(op, 2)->dyn_cast<paddle::dialect::FullOp>();
    if (!full_op) {
      VLOG(3) << "Can not find full op";
      return false;
    } else {
      auto axis = full_op->attribute<paddle::dialect::ScalarAttribute>("value")
                      .data()
                      .to<int>();
      auto x_shape = op.operand_source(0)
                         .type()
                         .dyn_cast<paddle::dialect::DenseTensorType>()
                         .dims();
      auto out_vector_type = op.result(0).type().dyn_cast<pir::VectorType>();

      paddle::dialect::FullIntArrayOp full_sections_op =
          pir::GetDefiningOpForInput(op, 1)
              ->dyn_cast<paddle::dialect::FullIntArrayOp>();
      if (!full_sections_op) {
        VLOG(3) << "Can not find FullIntArrayOp";
        return false;
      }

      auto sections = full_sections_op->attribute<pir::ArrayAttribute>("value");

      std::vector<int64_t> output_lengths;
      for (const auto &attr : sections.AsVector()) {
        output_lengths.push_back(attr.dyn_cast<pir::Int64Attribute>().data());
      }
      axis += (axis < 0) ? x_shape.size() : 0;

      if (x_shape[axis] == -1) {
        VLOG(3) << "The (" << axis << ") dim of input should not be -1";
        return false;
      }

      if (output_lengths.size() != out_vector_type.size()) {
        VLOG(3) << "The output_length should be equal to the output size.";
        return false;
      }
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};
class SplitWithNumOpPattern
    : public pir::OpRewritePattern<paddle::dialect::SplitWithNumOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::SplitWithNumOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::SplitWithNumOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    paddle::dialect::FullOp full_op =
        pir::GetDefiningOpForInput(op, 1)->dyn_cast<paddle::dialect::FullOp>();
    if (!full_op) {
      VLOG(3) << "Can not find full op";
      return false;
    } else {
      auto axis = full_op->attribute<paddle::dialect::ScalarAttribute>("value")
                      .data()
                      .to<int>();
      auto x_shape = op.operand_source(0)
                         .type()
                         .dyn_cast<paddle::dialect::DenseTensorType>()
                         .dims();
      auto out_vector_type = op.result(0).type().dyn_cast<pir::VectorType>();

      axis += (axis < 0) ? x_shape.size() : 0;
      if (x_shape[axis] == -1) {
        VLOG(3) << "The (" << axis << ") dim of input should not be -1";
        return false;
      }

      if (!op->HasAttribute("num")) {
        VLOG(3) << "split_with_num op must has num attributes";
        return false;
      }
      int num = op->attribute<pir::Int32Attribute>("num").data();
      std::vector<int64_t> output_lengths;
      if (num > 0) {
        int64_t in_axis_dim = x_shape[axis];
        if (in_axis_dim % num != 0) {
          VLOG(3) << "Invalid number to split. Tensor split does not result"
                     " in an equal division of dimensions. Axis dim = "
                  << in_axis_dim << " num = " << num << "!= 0";
          return false;
        }
        size_t out_axis_dim = in_axis_dim / num;
        for (int i = 0; i < num; ++i) {
          output_lengths.push_back(out_axis_dim);
        }
      }

      if (out_vector_type.size() != output_lengths.size()) {
        VLOG(3) << "The output_length should be equal to the output size.";
        return false;
      }
      op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
      return true;
    }
  }
};
class GreaterEqualOpPattern
    : public pir::OpRewritePattern<paddle::dialect::GreaterEqualOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::GreaterEqualOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::GreaterEqualOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
#if IS_TRT_VERSION_LT(8400)
    VLOG(3) << "GreaterEqualOp is not supported when TensorRT < 8.4";
    return false;
#else
    pir::Value x = op.operand_source(0);
    pir::Value y = op.operand_source(1);
    auto x_dtype = pir::GetDataTypeFromValue(x);
    auto y_dtype = pir::GetDataTypeFromValue(y);
    if (x_dtype.isa<pir::BoolType>() || y_dtype.isa<pir::BoolType>()) {
      VLOG(3) << "Greate_equal op do not support bool datatype";
      return false;
    }
#endif
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};
class MultiplyOpPattern
    : public pir::OpRewritePattern<paddle::dialect::MultiplyOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::MultiplyOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::MultiplyOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    pir::Value x = op.operand_source(0);
    pir::Value y = op.operand_source(1);
    auto x_dtype = pir::GetDataTypeFromValue(x);
    auto y_dtype = pir::GetDataTypeFromValue(y);
    if (x_dtype.isa<pir::BoolType>() || y_dtype.isa<pir::BoolType>()) {
      VLOG(3) << "elementwise_mul do not support boolean datatype.";
      return false;
    }

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
    ADD_PATTERN(Bmm)
    ADD_PATTERN(Concat)
    ADD_PATTERN(Full)
    ADD_PATTERN(Fused_gemm_epilogue)
    ADD_PATTERN(Add)
    ADD_PATTERN(Layer_norm)
    ADD_PATTERN(Silu)
    ADD_PATTERN(Conv2d)
    ADD_PATTERN(FusedConv2dAddAct)
    ADD_PATTERN(DepthwiseConv2d)
    ADD_PATTERN(Nonzero)
    ADD_PATTERN(Gelu)
    ADD_PATTERN(Sigmoid)

#undef ADD_PATTERN
    ps.Add(std::make_unique<Pool2dOpPattern>(context));
    ps.Add(std::make_unique<Conv2dTransposeOpPattern>(context));
    ps.Add(std::make_unique<DepthwiseConv2dTransposeOpPattern>(context));
    ps.Add(std::make_unique<DeformableConvOpPattern>(context));
    ps.Add(std::make_unique<ArangeOpPattern>(context));
    ps.Add(std::make_unique<SignOpPattern>(context));
    ps.Add(std::make_unique<LogicalNotOpPattern>(context));
    ps.Add(std::make_unique<GroupNormOpPattern>(context));
    ps.Add(std::make_unique<TransposeOpPattern>(context));
    ps.Add(std::make_unique<GatherOpPattern>(context));
    ps.Add(std::make_unique<GatherNdOpPattern>(context));
    ps.Add(std::make_unique<ScaleOpPattern>(context));
    ps.Add(std::make_unique<UnsqueezeOpPattern>(context));
    ps.Add(std::make_unique<SqueezeOpPattern>(context));
    ps.Add(std::make_unique<Unsqueeze_OpPattern>(context));
    ps.Add(std::make_unique<SliceOpPattern>(context));
    ps.Add(std::make_unique<IndexSelectOpPattern>(context));
    ps.Add(std::make_unique<FlattenOpPattern>(context));
    ps.Add(std::make_unique<CastOpPattern>(context));
    ps.Add(std::make_unique<SplitOpPattern>(context));
    ps.Add(std::make_unique<SplitWithNumOpPattern>(context));
    ps.Add(std::make_unique<GreaterEqualOpPattern>(context));
    ps.Add(std::make_unique<MultiplyOpPattern>(context));
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
