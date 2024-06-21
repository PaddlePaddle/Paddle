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

#include "paddle/fluid/pir/transforms/xpu/conv2d_add_fuse_pass.h"
#include <optional>

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/analysis_info.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace {

class Conv2dAddXpuFusePattern : public paddle::drr::DrrPatternBase {
 private:
  bool enable_int8_;
  const std::map<std::string, int> &quant_post_type_;
  std::reference_wrapper<std::optional<pir::detail::PassExecutionState>>
      pass_state_;

 public:
  explicit Conv2dAddXpuFusePattern(
      bool enable_int8,
      const std::map<std::string, int> &quant_post_type,
      std::reference_wrapper<std::optional<pir::detail::PassExecutionState>>
          pass_state)
      : enable_int8_(enable_int8),
        quant_post_type_(quant_post_type),
        pass_state_(pass_state) {}
  std::string name() const override { return "Conv2dAddFusePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    // bool enable_int8_ = false;
    // add constraint
    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      if (!this->pass_state_.get().has_value()) {
        VLOG(5) << "pass_state_ has no value";
      }
      if (pass_state_.get()
              ->preserved_analyses.IsPreserved<pir::pass::Int8Analysis>()) {
        auto &int8_analysis =
            pass_state_.get()->am.GetAnalysis<pir::pass::Int8Analysis>();
        if (enable_int8_ == int8_analysis.enable_int8)
          return true;
        else
          return false;
      }
      return true;
    });
    // conv2d
    const auto &conv2d =
        pat.Op(paddle::dialect::Conv2dOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});
    conv2d({&pat.Tensor("input"), &pat.Tensor("filter")},
           {&pat.Tensor("conv2d_out")});
    // add
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    pat.Tensor("add_out") = add(pat.Tensor("conv2d_out"), pat.Tensor("y"));
    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
      auto padding_algorithm = match_ctx.Attr<std::string>("padding_algorithm");
      if (padding_algorithm != "EXPLICIT" && padding_algorithm != "SAME" &&
          padding_algorithm != "VALID") {
        return false;
      }
      auto groups = match_ctx.Attr<int>("groups");
      if (groups < 1) {
        return false;
      }
      auto data_format = match_ctx.Attr<std::string>("data_format");
      if (data_format != "NCHW" && data_format != "AnyLayout") {
        return false;
      }
      std::vector<int64_t> conv_input_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("input"));
      auto paddings_size = match_ctx.Attr<std::vector<int>>("paddings");
      std::vector<int64_t> filter_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("filter"));
      std::vector<int64_t> y_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("y"));
      if (conv_input_shape.size() != 4) {
        return false;
      }
      if (!(paddings_size.size() == 2 || paddings_size.size() == 4)) {
        return false;
      }
      if (y_shape.size() != 4 || y_shape[1] != filter_shape[0] ||
          y_shape[0] * y_shape[2] * y_shape[3] != 1) {
        VLOG(5) << "Conv2dAddFusePattern y_shape false: ";
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    // scale filter
    const auto &scale_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> int32_t {
          return 32767;
        });
    // reshape scale shape
    const auto &expand_1_shape =
        res.ComputeAttr([&](const paddle::drr::MatchContext &match_ctx)
                            -> std::vector<int64_t> {
          return {static_cast<int64_t>(
              phi::backends::xpu::get_xpu_max_ptr_size(-1))};
        });

    // paddings
    const auto &paddings_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
          auto paddings = match_ctx.Attr<std::vector<int>>("paddings");
          if (paddings.size() == 2) {
            return {paddings[0], paddings[0], paddings[1], paddings[1]};
          } else {
            return paddings;
          }
        });

    const auto &out_dtype_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
          auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("input"));
          if (x_dtype.isa<pir::Float32Type>()) {
            return phi::DataType::FLOAT32;
          } else {
            return phi::DataType::UNDEFINED;
          }
        });

    // get max filter and max x
    const auto &abs_op = res.Op(paddle::dialect::AbsOp::name());
    const auto &max_op1 =
        res.Op(paddle::dialect::MaxOp::name(),
               {{"axis", res.VectorInt64Attr(std::vector<int64_t>{})},
                {"keepdim", res.BoolAttr(false)}});

    const auto &div = res.Op(paddle::dialect::DivideOp::name());

    // quant clip（x * qmax / float_max）
    res.Tensor("res_filter_max");
    res.Tensor("scale_max");
    res.Tensor("scale_fp32_max");
    if (!enable_int8_) {
      if (quant_post_type_.find("conv2d") != quant_post_type_.end() &&
              quant_post_type_.find("conv2d")->second == 2 ||
          quant_post_type_.find("conv2d") != quant_post_type_.end() &&
              quant_post_type_.find("conv2d")->second == -1) {
        VLOG(5) << "Use int16 per-tensor weight";
        // find fp32 filter_max
        res.Tensor("filter_abs") = abs_op(res.Tensor("filter"));
        res.Tensor("filter_fp_max") = max_op1(res.Tensor("filter_abs"));
        // set filter_max shape
        const auto &filter_max_shape_attr =
            res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx)
                                -> std::vector<int64_t> { return {1}; });
        // full 32767 or 127
        const auto &full1 = res.Op(paddle::dialect::FullOp::name(),
                                   {{"shape", filter_max_shape_attr},
                                    {"value", scale_attr},
                                    {"dtype", res.DataTypeAttr("float32")},
                                    {"place", res.PlaceAttr("cpu")}});
        // cast fp32->int16
        const auto &cast_op = res.Op(paddle::dialect::CastOp::name(),
                                     {{"dtype", res.DataTypeAttr("int16")}});
        // scale1 = filter_fp_max/32767
        res.Tensor("scale_fp32_max") =
            div(res.Tensor("filter_fp_max"), full1());
        // int16_weight = filter / scale1
        res.Tensor("filter_quant_tmp") =
            div(res.Tensor("filter"), res.Tensor("scale_fp32_max"));
        // cast fp32->int16
        res.Tensor("filter_quant") = cast_op(res.Tensor("filter_quant_tmp"));
        // expand to 4 or 6 pointer size
        const auto &expand = res.Op(paddle::dialect::ExpandOp::name(),
                                    {{"shape", expand_1_shape}});
        res.Tensor("res_filter_max") = expand(res.Tensor("filter_fp_max"));
      } else if (quant_post_type_.find("conv2d") != quant_post_type_.end() &&
                 quant_post_type_.find("conv2d")->second == 3) {
        VLOG(5) << "Unsupported int16 per-channel weight";

      } else if (quant_post_type_.find("conv2d") != quant_post_type_.end() &&
                 quant_post_type_.find("conv2d")->second == 4) {
        VLOG(5) << "Use int31 per-tensor weight";  // todo not quant
      } else if (quant_post_type_.find("conv2d") != quant_post_type_.end() &&
                     quant_post_type_.find("conv2d")->second == 0 ||
                 quant_post_type_.find("conv2d") != quant_post_type_.end() &&
                     quant_post_type_.find("conv2d")->second == 1) {
        VLOG(5) << "Unsupported int8 post quant !";
      } else {
        VLOG(5) << "Unsupported type weight by non-int8!";
      }
    } else {
      VLOG(5) << "Use int8 quant weight";
      // TODO(gitliuyf): int8 per-tensor
      VLOG(5) << "Unsupported int8 post quant !";
    }

    // fusion op, set max x nullptr
    const auto &fused_conv2d_add_act =
        res.Op(paddle::dialect::Conv2dXpuOp::name(),
               {{
                   {"paddings", paddings_attr},
                   {"dilations", pat.Attr("dilations")},
                   {"strides", pat.Attr("strides")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"groups", pat.Attr("groups")},
                   {"act_type",
                    res.Int32Attr(static_cast<int>(xpu::Activation_t::LINEAR))},
                   {"act_param", res.Float32Attr(0.0f)},
                   {"out_dtype", out_dtype_attr},
               }});

    fused_conv2d_add_act(
        {
            &res.Tensor("input"),
            &res.InputNoneTensor(),
            &res.Tensor("filter_quant"),
            &res.Tensor("res_filter_max"),
            &res.Tensor("y"),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.Tensor("scale_fp32_max"),
        },
        {
            &res.Tensor("add_out"),
            &res.Tensor("out_max"),
        });
  }
};

class Conv2dAddFusePass : public pir::PatternRewritePass {
 public:
  Conv2dAddFusePass() : pir::PatternRewritePass("conv2d_add_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

    std::map<std::string, int> default_type;
    default_type.insert(std::make_pair("conv2d", -1));
    const std::map<std::string, int> quant_post_type =
        Has("quant_post_dynamic_weight_methods")
            ? Get<std::map<std::string, int>>(
                  "quant_post_dynamic_weight_methods")
            : default_type;
    for (auto it = quant_post_type.begin(); it != quant_post_type.end(); ++it) {
      VLOG(5) << "Key:" << it->first;
      VLOG(5) << "Value:" << it->second;
    }

    // TODO(gitliuyf): int8 quant
    // ps.Add(paddle::drr::Create<Conv2dAddXpuFusePattern>(
    //     context, true, quant_post_type, std::ref(pass_state())));
    ps.Add(paddle::drr::Create<Conv2dAddXpuFusePattern>(
        context, false, quant_post_type, std::ref(pass_state())));
    return ps;
  }
};

}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateConv2dAddXpuFusePass() {
  return std::make_unique<Conv2dAddFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(conv2d_add_xpu_fuse_pass, Conv2dAddFusePass);
