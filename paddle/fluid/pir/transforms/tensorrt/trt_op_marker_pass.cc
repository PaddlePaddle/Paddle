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
    auto padding_attr = op->attribute<pir::ArrayAttribute>("padding");
    std::vector<int> paddings;
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
            paddle::dialect::FullIntArrayOp full_int_array_op =
                pir::GetDefiningOpForInput(op, 1)
                    ->dyn_cast<paddle::dialect::FullIntArrayOp>();
            if (!full_int_array_op) {
              VLOG(3) << "Cannot find FullIntArrayOp";
              return false;
            } else {
              auto attr_value =
                  full_int_array_op->attribute<pir::ArrayAttribute>("value");
              std::vector<int> kernel_size;
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
      return false;
    }
#if IS_TRT_VERSION_LT(7000)
    if (op->HasAttribute("padding_algorithm")) {
      std::string padding_algorithm =
          op->attribute<pir::StrAttribute>("padding_algorithm").AsString();
      if (padding_algorithm == "SAME" && op->HasAttribute("strides")) {
        auto strides_attr = op->attribute<pir::ArrayAttribute>("strides");
        std::vector<int> strides;
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
#undef ADD_PATTERN
    ps.Add(std::make_unique<Pool2dOpPattern>(context));
    ps.Add(std::make_unique<Conv2dOpPattern>(context));
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
