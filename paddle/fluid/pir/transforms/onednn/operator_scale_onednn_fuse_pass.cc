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

#include "paddle/fluid/pir/transforms/onednn/operator_scale_onednn_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {
class OperatorScaleFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string fusable_ops_;
  std::string fused_ops_name_;
  uint32_t benefit_;

 public:
  OperatorScaleFusePattern(const std::string &fusable_ops,
                           const std::string &fused_ops_name,
                           uint32_t benefit)
      : fusable_ops_(fusable_ops),
        fused_ops_name_(fused_ops_name),
        benefit_(benefit) {}

  std::string name() const override {
    return fusable_ops_ + "ScaleFusePattern";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;

    if (fusable_ops_ == paddle::onednn::dialect::FcOp::name()) {
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
      op_attrs.emplace("fused_reshape2_shape",
                       pat.Attr("fused_reshape2_shape"));
    } else if (fusable_ops_ == paddle::dialect::MatmulOp::name()) {
      op_attrs.emplace("transpose_x", pat.Attr("transpose_x"));
      op_attrs.emplace("transpose_y", pat.Attr("transpose_y"));
    } else if (fusable_ops_ == paddle::onednn::dialect::FusedMatmulOp::name()) {
      op_attrs.emplace("trans_x", pat.Attr("trans_x"));
      op_attrs.emplace("trans_y", pat.Attr("trans_y"));
      op_attrs.emplace("matmul_alpha", pat.Attr("matmul_alpha"));
      op_attrs.emplace("fuse_activation", pat.Attr("fuse_activation"));
      op_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
      op_attrs.emplace("fuse_beta", pat.Attr("fuse_beta"));
      op_attrs.emplace("fused_output_scale", pat.Attr("fused_output_scale"));
      op_attrs.emplace("fused_reshape_x", pat.Attr("fused_reshape_x"));
      op_attrs.emplace("fused_transpose_x", pat.Attr("fused_transpose_x"));
      op_attrs.emplace("fused_reshape_y", pat.Attr("fused_reshape_y"));
      op_attrs.emplace("fused_transpose_y", pat.Attr("fused_transpose_y"));
      op_attrs.emplace("fused_reshape_out", pat.Attr("fused_reshape_out"));
      op_attrs.emplace("fused_transpose_out", pat.Attr("fused_transpose_out"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("scale_x", pat.Attr("scale_x"));
      op_attrs.emplace("scale_y", pat.Attr("scale_y"));
      op_attrs.emplace("scale_in_eltwise", pat.Attr("scale_in_eltwise"));
      op_attrs.emplace("scale_out", pat.Attr("scale_out"));
      op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));
    } else if (fusable_ops_ ==
                   paddle::onednn::dialect::FusedElementwiseAddOp::name() ||
               fusable_ops_ ==
                   paddle::onednn::dialect::FusedElementwiseSubOp::name() ||
               fusable_ops_ ==
                   paddle::onednn::dialect::FusedElementwiseMulOp::name() ||
               fusable_ops_ ==
                   paddle::onednn::dialect::FusedElementwiseDivOp::name()) {
      op_attrs.emplace("axis", pat.Attr("axis"));
      op_attrs.emplace("fuse_activation", pat.Attr("fuse_activation"));
      op_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
      op_attrs.emplace("fuse_beta", pat.Attr("fuse_beta"));
      op_attrs.emplace("fused_output_scale", pat.Attr("fused_output_scale"));
      op_attrs.emplace("fused_unsqueeze2_axes",
                       pat.Attr("fused_unsqueeze2_axes"));
      op_attrs.emplace("scale_x", pat.Attr("scale_x"));
      op_attrs.emplace("scale_y", pat.Attr("scale_y"));
      op_attrs.emplace("scale_out", pat.Attr("scale_out"));
    }

    const auto &op = pat.Op(fusable_ops_, op_attrs);

    const auto &full_1 = pat.Op(paddle::dialect::FullOp::name(),
                                {{"value", pat.Attr("full_1_value")}});
    pat.Tensor("Scale") = full_1();
    const auto &scale =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{
                   {"bias", pat.Attr("bias")},
                   {"bias_after_scale", pat.Attr("bias_after_scale")},
               }});
    if (fusable_ops_ == paddle::onednn::dialect::FcOp::name() ||
        fusable_ops_ == paddle::onednn::dialect::FusedMatmulOp::name()) {
      op({&pat.Tensor("X"), &pat.Tensor("Y"), &pat.Tensor("Input3")},
         {&pat.Tensor("Out")});
    } else {
      op({&pat.Tensor("X"), &pat.Tensor("Y")}, {&pat.Tensor("Out")});
    }

    pat.Tensor("Scale_out") = scale(pat.Tensor("Out"), pat.Tensor("Scale"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto scale_bias = match_ctx.Attr<float>("bias");
      if (scale_bias != 0.0) {
        return false;
      }
      return true;
    });

    if (fusable_ops_ == paddle::onednn::dialect::FusedMatmulOp::name() ||
        fusable_ops_ ==
            paddle::onednn::dialect::FusedElementwiseAddOp::name() ||
        fusable_ops_ ==
            paddle::onednn::dialect::FusedElementwiseSubOp::name() ||
        fusable_ops_ ==
            paddle::onednn::dialect::FusedElementwiseMulOp::name() ||
        fusable_ops_ ==
            paddle::onednn::dialect::FusedElementwiseDivOp::name()) {
      pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
        auto fused_output_scale = match_ctx.Attr<float>("fused_output_scale");
        if (fused_output_scale != 1.0) {
          // It means that it has been fused and has a value.
          return false;
        }
        return true;
      });
    }

    paddle::drr::ResultPattern res = pat.ResultPattern();
    std::unordered_map<std::string, paddle::drr::Attribute> fused_op_attrs{};

    const auto &fused_output_scale = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<double>("full_1_value");
        });
    if (fusable_ops_ == paddle::onednn::dialect::FcOp::name()) {
      fused_op_attrs.emplace("in_num_col_dims", pat.Attr("in_num_col_dims"));
      fused_op_attrs.emplace("activation_type", pat.Attr("activation_type"));
      fused_op_attrs.emplace("padding_weights", pat.Attr("padding_weights"));
      fused_op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));
      fused_op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      fused_op_attrs.emplace("scale_in", pat.Attr("scale_in"));
      fused_op_attrs.emplace("scale_weights", pat.Attr("scale_weights"));
      fused_op_attrs.emplace("scale_out", pat.Attr("scale_out"));
      fused_op_attrs.emplace("force_fp32_output",
                             pat.Attr("force_fp32_output"));
      fused_op_attrs.emplace("fuse_activation", pat.Attr("fuse_activation"));
      fused_op_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
      fused_op_attrs.emplace("fuse_beta", pat.Attr("fuse_beta"));
      fused_op_attrs.emplace("fused_output_scale", fused_output_scale);
      fused_op_attrs.emplace("fused_reshape2_shape",
                             pat.Attr("fused_reshape2_shape"));

    } else if (fusable_ops_ == paddle::onednn::dialect::FusedMatmulOp::name()) {
      fused_op_attrs.emplace("matmul_alpha", pat.Attr("matmul_alpha"));
      fused_op_attrs.emplace("fuse_activation", pat.Attr("fuse_activation"));
      fused_op_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
      fused_op_attrs.emplace("fuse_beta", pat.Attr("fuse_beta"));
      fused_op_attrs.emplace("fused_output_scale", fused_output_scale);
      fused_op_attrs.emplace("fused_reshape_x", pat.Attr("fused_reshape_x"));
      fused_op_attrs.emplace("fused_transpose_x",
                             pat.Attr("fused_transpose_x"));
      fused_op_attrs.emplace("fused_reshape_y", pat.Attr("fused_reshape_y"));
      fused_op_attrs.emplace("fused_transpose_y",
                             pat.Attr("fused_transpose_y"));
      fused_op_attrs.emplace("fused_reshape_out",
                             pat.Attr("fused_reshape_out"));
      fused_op_attrs.emplace("fused_transpose_out",
                             pat.Attr("fused_transpose_out"));
      fused_op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      fused_op_attrs.emplace("scale_x", pat.Attr("scale_x"));
      fused_op_attrs.emplace("scale_y", pat.Attr("scale_y"));
      fused_op_attrs.emplace("scale_in_eltwise", pat.Attr("scale_in_eltwise"));
      fused_op_attrs.emplace("scale_out", pat.Attr("scale_out"));
      fused_op_attrs.emplace("trans_x", pat.Attr("trans_x"));
      fused_op_attrs.emplace("trans_y", pat.Attr("trans_y"));
      fused_op_attrs.emplace("force_fp32_output",
                             pat.Attr("force_fp32_output"));

    } else if (fusable_ops_ == paddle::dialect::MatmulOp::name()) {
      fused_op_attrs.emplace("trans_x", pat.Attr("transpose_x"));
      fused_op_attrs.emplace("trans_y", pat.Attr("transpose_y"));
      fused_op_attrs.emplace("matmul_alpha", res.Float32Attr(1.0f));
      fused_op_attrs.emplace("fuse_activation", res.StrAttr(""));
      fused_op_attrs.emplace("fuse_alpha", res.Float32Attr(0.0f));
      fused_op_attrs.emplace("fuse_beta", res.Float32Attr(0.0f));
      fused_op_attrs.emplace("fused_output_scale", fused_output_scale);
      fused_op_attrs.emplace("fused_reshape_x", res.VectorInt32Attr({}));
      fused_op_attrs.emplace("fused_transpose_x", res.VectorInt32Attr({}));
      fused_op_attrs.emplace("fused_reshape_y", res.VectorInt32Attr({}));
      fused_op_attrs.emplace("fused_transpose_y", res.VectorInt32Attr({}));
      fused_op_attrs.emplace("fused_reshape_out", res.VectorInt32Attr({}));
      fused_op_attrs.emplace("fused_transpose_out", res.VectorInt32Attr({}));
      fused_op_attrs.emplace("mkldnn_data_type", res.StrAttr("float32"));
      fused_op_attrs.emplace("scale_x", res.Float32Attr(1.0f));
      fused_op_attrs.emplace("scale_y", res.Float32Attr(1.0f));
      fused_op_attrs.emplace("scale_in_eltwise", res.Float32Attr(0.0f));
      fused_op_attrs.emplace("scale_out", res.Float32Attr(1.0f));
      fused_op_attrs.emplace("force_fp32_output", res.BoolAttr(false));

    } else if (fusable_ops_ ==
                   paddle::onednn::dialect::FusedElementwiseAddOp::name() ||
               fusable_ops_ ==
                   paddle::onednn::dialect::FusedElementwiseSubOp::name() ||
               fusable_ops_ ==
                   paddle::onednn::dialect::FusedElementwiseMulOp::name() ||
               fusable_ops_ ==
                   paddle::onednn::dialect::FusedElementwiseDivOp::name()) {
      fused_op_attrs.emplace("axis", pat.Attr("axis"));
      fused_op_attrs.emplace("fuse_activation", pat.Attr("fuse_activation"));
      fused_op_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
      fused_op_attrs.emplace("fuse_beta", pat.Attr("fuse_beta"));
      fused_op_attrs.emplace("fused_output_scale", fused_output_scale);
      fused_op_attrs.emplace("fused_unsqueeze2_axes",
                             pat.Attr("fused_unsqueeze2_axes"));
      fused_op_attrs.emplace("scale_x", pat.Attr("scale_x"));
      fused_op_attrs.emplace("scale_y", pat.Attr("scale_y"));
      fused_op_attrs.emplace("scale_out", pat.Attr("scale_out"));
    } else {
      // Add, Sub, Mul, Div
      fused_op_attrs.emplace("axis", res.Int32Attr(-1));
      fused_op_attrs.emplace("fuse_activation", res.StrAttr(""));
      fused_op_attrs.emplace("fuse_alpha", res.Float32Attr(0.0f));
      fused_op_attrs.emplace("fuse_beta", res.Float32Attr(0.0f));
      fused_op_attrs.emplace("fused_output_scale", fused_output_scale);
      fused_op_attrs.emplace("fused_unsqueeze2_axes", res.VectorInt32Attr({}));
      fused_op_attrs.emplace("scale_x", res.Float32Attr(1.0f));
      fused_op_attrs.emplace("scale_y", res.Float32Attr(1.0f));
      fused_op_attrs.emplace("scale_out", res.Float32Attr(1.0f));
    }

    const auto &fused_op = res.Op(fused_ops_name_, fused_op_attrs);
    if (fusable_ops_ == paddle::onednn::dialect::FcOp::name() ||
        fusable_ops_ == paddle::onednn::dialect::FusedMatmulOp::name()) {
      fused_op({&res.Tensor("X"), &res.Tensor("Y"), &res.Tensor("Input3")},
               {&res.Tensor("Scale_out")});
    } else if (fusable_ops_ == paddle::dialect::MatmulOp::name()) {
      fused_op({&res.Tensor("X"), &res.Tensor("Y"), &res.InputNoneTensor()},
               {&res.Tensor("Scale_out")});
    } else {
      fused_op({&res.Tensor("X"), &res.Tensor("Y")},
               {&res.Tensor("Scale_out")});
    }
  }
};

class OperatorScaleFusePass : public pir::PatternRewritePass {
 public:
  OperatorScaleFusePass()
      : pir::PatternRewritePass("operator_scale_onednn_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    const std::vector<std::string> fusable_ops{
        paddle::onednn::dialect::FcOp::name(),
        paddle::dialect::MatmulOp::name(),
        paddle::onednn::dialect::FusedMatmulOp::name(),
        paddle::onednn::dialect::FusedElementwiseAddOp::name(),
        paddle::onednn::dialect::FusedElementwiseSubOp::name(),
        paddle::onednn::dialect::FusedElementwiseMulOp::name(),
        paddle::onednn::dialect::FusedElementwiseDivOp::name(),
        paddle::dialect::AddOp::name(),
        paddle::dialect::SubtractOp::name(),
        paddle::dialect::MultiplyOp::name(),
        paddle::dialect::DivideOp::name(),
    };

    const std::vector<std::string> fused_ops{
        paddle::onednn::dialect::FcOp::name(),
        paddle::onednn::dialect::FusedMatmulOp::name(),
        paddle::onednn::dialect::FusedMatmulOp::name(),
        paddle::onednn::dialect::FusedElementwiseAddOp::name(),
        paddle::onednn::dialect::FusedElementwiseSubOp::name(),
        paddle::onednn::dialect::FusedElementwiseMulOp::name(),
        paddle::onednn::dialect::FusedElementwiseDivOp::name(),
        paddle::onednn::dialect::FusedElementwiseAddOp::name(),
        paddle::onednn::dialect::FusedElementwiseSubOp::name(),
        paddle::onednn::dialect::FusedElementwiseMulOp::name(),
        paddle::onednn::dialect::FusedElementwiseDivOp::name(),
    };
    int benefit_idx = 1;
    int fused = 0;
    for (auto op : fusable_ops) {
      ps.Add(paddle::drr::Create<OperatorScaleFusePattern>(
          context, op, fused_ops[fused++], benefit_idx));
      benefit_idx++;
    }

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateOperatorScaleFusePass() {
  return std::make_unique<OperatorScaleFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(operator_scale_onednn_fuse_pass, OperatorScaleFusePass);
