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
#include "paddle/fluid/pir/transforms/onednn/operator_reshape_onednn_fuse_pass.h"
#include <glog/logging.h>

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {
class FusedTransposeReshapeFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string fusable_ops_;
  std::string fused_ops_name_;
  uint32_t benefit_;

 public:
  FusedTransposeReshapeFusePattern(const std::string &fusable_ops,
                                   const std::string &fused_ops_name,
                                   uint32_t benefit)
      : fusable_ops_(fusable_ops),
        fused_ops_name_(fused_ops_name),
        benefit_(benefit) {}

  std::string name() const override {
    return fusable_ops_ + "ReshapeFusePattern";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    op_attrs.emplace("axis", pat.Attr("axis"));
    op_attrs.emplace("fused_squeeze2_axes", pat.Attr("fused_squeeze2_axes"));
    op_attrs.emplace("fused_unsqueeze2_axes",
                     pat.Attr("fused_unsqueeze2_axes"));
    op_attrs.emplace("fused_reshape2_shape", pat.Attr("fused_reshape2_shape"));
    op_attrs.emplace("scale", pat.Attr("scale"));
    op_attrs.emplace("shift", pat.Attr("shift"));
    op_attrs.emplace("output_data_type", pat.Attr("output_data_type"));
    op_attrs.emplace("data_format", pat.Attr("data_format"));
    op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));

    const auto &op = pat.Op(fusable_ops_, op_attrs);

    const auto &full_1 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_1_value")}});

    pat.Tensor("shape") = full_1();

    const auto &reshape = pat.Op(paddle::dialect::ReshapeOp::name());

    op({&pat.Tensor("X")}, {&pat.Tensor("Out")});

    reshape({&pat.Tensor("Out"), &pat.Tensor("shape")},
            {&pat.Tensor("ShapeOut")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      int num_of_minus_ones = 0;
      auto reshape2_shape =
          match_ctx.Attr<std::vector<int64_t>>("full_1_value");

      for (auto item : reshape2_shape) {
        if (item == 0) {
          VLOG(4) << "OneDNN op+reshape2 fuse pass does not support zero dims";
          return false;
        } else if (item == -1) {
          ++num_of_minus_ones;
        }
      }

      if (num_of_minus_ones > 1) {
        VLOG(4) << "Number of -1 values inside of reshape2 should be 0 or 1";
        return false;
      }
      return true;
    });

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto fused_unsqueeze2_axes =
          match_ctx.Attr<std::vector<int>>("fused_unsqueeze2_axes");
      auto fused_reshape2_shape =
          match_ctx.Attr<std::vector<int>>("fused_reshape2_shape");

      if (fused_unsqueeze2_axes.size() > 0 || fused_reshape2_shape.size()) {
        VLOG(4) << fusable_ops_ << " is already fused with unsqueeze/reshape";
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    std::unordered_map<std::string, paddle::drr::Attribute> fused_op_attrs{};

    const auto &fused_reshape2_shape = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
          std::vector<int> int_array_value;
          auto shape = match_ctx.Attr<std::vector<int64_t>>("full_1_value");
          for (auto i : shape) {
            int_array_value.emplace_back(static_cast<int>(i));
          }
          return int_array_value;
        });

    fused_op_attrs.emplace("axis", pat.Attr("axis"));
    fused_op_attrs.emplace("fused_squeeze2_axes",
                           pat.Attr("fused_squeeze2_axes"));
    fused_op_attrs.emplace("fused_unsqueeze2_axes",
                           pat.Attr("fused_unsqueeze2_axes"));
    fused_op_attrs.emplace("fused_reshape2_shape", fused_reshape2_shape);
    fused_op_attrs.emplace("scale", pat.Attr("scale"));
    fused_op_attrs.emplace("shift", pat.Attr("shift"));
    fused_op_attrs.emplace("output_data_type", pat.Attr("output_data_type"));
    fused_op_attrs.emplace("data_format", pat.Attr("data_format"));
    fused_op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));

    const auto &fused_op = res.Op(fused_ops_name_, fused_op_attrs);

    fused_op({&res.Tensor("X")}, {&res.Tensor("ShapeOut")});
  }
};

class FcReshapeFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string fusable_ops_;
  std::string fused_ops_name_;
  uint32_t benefit_;

 public:
  FcReshapeFusePattern(const std::string &fusable_ops,
                       const std::string &fused_ops_name,
                       uint32_t benefit)
      : fusable_ops_(fusable_ops),
        fused_ops_name_(fused_ops_name),
        benefit_(benefit) {}

  std::string name() const override {
    return fusable_ops_ + "ReshapeFusePattern";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;

    op_attrs.emplace("in_num_col_dims", pat.Attr("in_num_col_dims"));
    op_attrs.emplace("activation_type", pat.Attr("activation_type"));
    op_attrs.emplace("padding_weights", pat.Attr("padding_weights"));
    op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));
    op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    op_attrs.emplace("scale_in", pat.Attr("scale_in"));
    op_attrs.emplace("scale_weights", pat.Attr("scale_weights"));
    op_attrs.emplace("scale_out", pat.Attr("scale_out"));
    op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));
    op_attrs.emplace("fuse_activation", pat.Attr("fuse_activation"));
    op_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
    op_attrs.emplace("fuse_beta", pat.Attr("fuse_beta"));
    op_attrs.emplace("fused_output_scale", pat.Attr("fused_output_scale"));
    op_attrs.emplace("fused_reshape2_shape", pat.Attr("fused_reshape2_shape"));

    const auto &op = pat.Op(fusable_ops_, op_attrs);

    const auto &full_1 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_1_value")}});

    pat.Tensor("shape") = full_1();

    const auto &reshape = pat.Op(paddle::dialect::ReshapeOp::name());

    op({&pat.Tensor("X"), &pat.Tensor("Y"), &pat.Tensor("Input3")},
       {&pat.Tensor("Out")});

    reshape({&pat.Tensor("Out"), &pat.Tensor("shape")},
            {&pat.Tensor("ShapeOut")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      int num_of_minus_ones = 0;
      auto reshape2_shape =
          match_ctx.Attr<std::vector<int64_t>>("full_1_value");

      for (auto item : reshape2_shape) {
        if (item == 0) {
          VLOG(4) << "OneDNN op+reshape2 fuse pass does not support zero dims";
          return false;
        } else if (item == -1) {
          ++num_of_minus_ones;
        }
      }

      if (num_of_minus_ones > 1) {
        VLOG(4) << "Number of -1 values inside of reshape2 should be 0 or 1";
        return false;
      }
      return true;
    });

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto fused_reshape2_shape =
          match_ctx.Attr<std::vector<int>>("fused_reshape2_shape");

      if (fused_reshape2_shape.size()) {
        VLOG(4) << fusable_ops_ << " is already fused with reshape";
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    std::unordered_map<std::string, paddle::drr::Attribute> fused_op_attrs{};

    const auto &fused_reshape2_shape = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
          std::vector<int> int_array_value;
          auto shape = match_ctx.Attr<std::vector<int64_t>>("full_1_value");
          for (auto i : shape) {
            int_array_value.emplace_back(static_cast<int>(i));
          }
          return int_array_value;
        });

    fused_op_attrs.emplace("in_num_col_dims", pat.Attr("in_num_col_dims"));
    fused_op_attrs.emplace("activation_type", pat.Attr("activation_type"));
    fused_op_attrs.emplace("padding_weights", pat.Attr("padding_weights"));
    fused_op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));
    fused_op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    fused_op_attrs.emplace("scale_in", pat.Attr("scale_in"));
    fused_op_attrs.emplace("scale_weights", pat.Attr("scale_weights"));
    fused_op_attrs.emplace("scale_out", pat.Attr("scale_out"));
    fused_op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));
    fused_op_attrs.emplace("fuse_activation", pat.Attr("fuse_activation"));
    fused_op_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
    fused_op_attrs.emplace("fuse_beta", pat.Attr("fuse_beta"));
    fused_op_attrs.emplace("fused_output_scale",
                           pat.Attr("fused_output_scale"));
    fused_op_attrs.emplace("fused_reshape2_shape", fused_reshape2_shape);

    const auto &fused_op = res.Op(fused_ops_name_, fused_op_attrs);

    fused_op({&res.Tensor("X"), &res.Tensor("Y"), &res.Tensor("Input3")},
             {&res.Tensor("ShapeOut")});
  }
};

class TransposeReshapeFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string fusable_ops_;
  std::string fused_ops_name_;
  uint32_t benefit_;

 public:
  TransposeReshapeFusePattern(const std::string &fusable_ops,
                              const std::string &fused_ops_name,
                              uint32_t benefit)
      : fusable_ops_(fusable_ops),
        fused_ops_name_(fused_ops_name),
        benefit_(benefit) {}

  std::string name() const override {
    return fusable_ops_ + "ReshapeFusePattern";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &op = pat.Op(fusable_ops_, {{"perm", pat.Attr("axis")}});

    const auto &full_1 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_1_value")}});

    pat.Tensor("shape") = full_1();

    const auto &reshape = pat.Op(paddle::dialect::ReshapeOp::name());

    op({&pat.Tensor("X")}, {&pat.Tensor("Out")});

    reshape({&pat.Tensor("Out"), &pat.Tensor("shape")},
            {&pat.Tensor("ShapeOut")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      int num_of_minus_ones = 0;
      auto reshape2_shape =
          match_ctx.Attr<std::vector<int64_t>>("full_1_value");

      for (auto item : reshape2_shape) {
        if (item == 0) {
          VLOG(4) << "OneDNN op+reshape2 fuse pass does not support zero dims";
          return false;
        } else if (item == -1) {
          ++num_of_minus_ones;
        }
      }

      if (num_of_minus_ones > 1) {
        VLOG(4) << "Number of -1 values inside of reshape2 should be 0 or 1";
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    std::unordered_map<std::string, paddle::drr::Attribute> fused_op_attrs{};

    const auto &fused_reshape2_shape = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
          std::vector<int> int_array_value;
          auto shape = match_ctx.Attr<std::vector<int64_t>>("full_1_value");
          for (auto i : shape) {
            int_array_value.emplace_back(static_cast<int>(i));
          }
          return int_array_value;
        });

    fused_op_attrs.emplace("axis", pat.Attr("axis"));
    fused_op_attrs.emplace("fused_squeeze2_axes", res.VectorInt32Attr({}));
    fused_op_attrs.emplace("fused_unsqueeze2_axes", res.VectorInt32Attr({}));
    fused_op_attrs.emplace("fused_reshape2_shape", fused_reshape2_shape);
    fused_op_attrs.emplace("scale", res.Float32Attr(1.0f));
    fused_op_attrs.emplace("shift", res.Float32Attr(0.0f));
    fused_op_attrs.emplace("output_data_type", res.StrAttr(""));
    fused_op_attrs.emplace("data_format", res.StrAttr("AnyLayout"));
    fused_op_attrs.emplace("mkldnn_data_type", res.StrAttr("float32"));

    const auto &fused_op = res.Op(fused_ops_name_, fused_op_attrs);

    fused_op({&res.Tensor("X")}, {&res.Tensor("ShapeOut")});
  }
};

class OperatorReshapePass : public pir::PatternRewritePass {
 public:
  OperatorReshapePass()
      : pir::PatternRewritePass("operator_reshape_onednn_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    int benefit_idx = 1;

    ps.Add(paddle::drr::Create<FcReshapeFusePattern>(
        context,
        paddle::onednn::dialect::FcOp::name(),
        paddle::onednn::dialect::FcOp::name(),
        benefit_idx++));

    ps.Add(paddle::drr::Create<FusedTransposeReshapeFusePattern>(
        context,
        paddle::onednn::dialect::FusedTransposeOp::name(),
        paddle::onednn::dialect::FusedTransposeOp::name(),
        benefit_idx++));

    ps.Add(paddle::drr::Create<TransposeReshapeFusePattern>(
        context,
        paddle::dialect::TransposeOp::name(),
        paddle::onednn::dialect::FusedTransposeOp::name(),
        benefit_idx++));

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateOperatorReshapeOneDNNPass() {
  return std::make_unique<OperatorReshapePass>();
}

}  // namespace pir

REGISTER_IR_PASS(operator_reshape_onednn_fuse_pass, OperatorReshapePass);
