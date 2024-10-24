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

#include "paddle/fluid/pir/transforms/gpu/matmul_add_act_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

std::set<std::string> act_ops = {
    "gelu",
    "relu",
};
std::unordered_map<std::string, std::string> activation_type = {
    {"gelu", paddle::dialect::GeluOp::name()},
    {"relu", paddle::dialect::ReluOp::name()},
};

class MatmulAddPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string fused_op_name_;
  bool reverse_add_;

 public:
  explicit MatmulAddPattern(const std::string &fused_op_name,
                            const bool reverse_add)
      : fused_op_name_(fused_op_name), reverse_add_(reverse_add) {}

  uint32_t benefit() const override {
    return fused_op_name_ == paddle::dialect::GemmEpilogueOp::name() ? 2 : 1;
  }
  std::string name() const override { return "MatmulAddPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("transpose_x")},
                                 {"transpose_y", pat.Attr("transpose_y")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    matmul({&pat.Tensor("x"), &pat.Tensor("w")}, {&pat.Tensor("matmul_out")});
    pat.Tensor("add_out") =
        reverse_add_ ? add(pat.Tensor("y"), pat.Tensor("matmul_out"))
                     : add(pat.Tensor("matmul_out"), pat.Tensor("y"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto w_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("w"));
      if (fused_op_name_ == paddle::dialect::GemmEpilogueOp::name()) {
        if (!w_dtype.isa<pir::Float16Type>() &&
            !w_dtype.isa<pir::BFloat16Type>()) {
          return false;
        }
      } else {
        if (!w_dtype.isa<pir::Float16Type>() &&
            !w_dtype.isa<pir::Float32Type>() &&
            !w_dtype.isa<pir::Float64Type>()) {
          return false;
        }
      }
      auto w_dims = pir::GetShapeFromValue(match_ctx.Tensor("w"));
      auto x_dims = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      auto y_dims = pir::GetShapeFromValue(match_ctx.Tensor("y"));
      if (w_dims.size() != 2 || x_dims.size() < 2) {
        return false;
      }
      // Currently，FcOp and GemmEpilogueOp support only RRR format
      if (x_dims.at(x_dims.size() - 1) != w_dims.at(0) ||
          match_ctx.Attr<bool>("transpose_x") == true ||
          match_ctx.Attr<bool>("transpose_y") == true) {
        return false;
      }

      // gemm_epilogue kernel requires gemm's N and K to be 8 aligned.
      // K and N correspond to w_dims[0] and w_dims[1] respectively.
      constexpr int cutlass_align = 8;
      if (fused_op_name_ == paddle::dialect::GemmEpilogueOp::name() &&
          (w_dims[0] % cutlass_align != 0 || w_dims[1] % cutlass_align != 0)) {
        return false;
      }

      if (y_dims.size() == 1) {
        return y_dims.at(0) == w_dims.at(1);
      }

      if (fused_op_name_ == paddle::dialect::FcOp::name()) {
        if (y_dims.size() == 2) {
          return y_dims.at(0) == 1 && y_dims.at(1) == w_dims.at(1);
        }
      } else {
        if (y_dims.size() == x_dims.size()) {
          if (y_dims.size() == 2) {
            return ((y_dims.at(0) == 1) || (y_dims.at(0) == x_dims.at(0))) &&
                   y_dims.at(1) == w_dims.at(1);
          }
          for (size_t ii = 0; ii < x_dims.size() - 1; ii++) {
            if (y_dims.at(ii) != x_dims.at(ii)) {
              return false;
            }
          }
          return y_dims.at(y_dims.size() - 1) == w_dims.at(1);
        }
      }
      return false;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &in_num_col_dims_attr =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> int {
          auto x_dims = pir::GetShapeFromValue(match_ctx.Tensor("x"));
          return static_cast<int>(x_dims.size()) - 1;
        });
    const auto &gemm_epilogue =
        res.Op(fused_op_name_,
               {{
                   {"in_num_col_dims", in_num_col_dims_attr},
                   {"activation_type", res.StrAttr("")},
                   {"padding_weights", res.BoolAttr(false)},
               }});
    gemm_epilogue({&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("y")},
                  {&res.Tensor("add_out")});
  }
};

// Act supports [relu, gelu]
class MatmulAddActPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string act_type_;
  std::string fused_op_name_;

 public:
  explicit MatmulAddActPattern(const std::string &act_type,
                               const std::string &fused_op_name)
      : act_type_(act_type), fused_op_name_(fused_op_name) {}
  uint32_t benefit() const override { return 3; }

  std::string name() const override { return "MatmulAddActPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &gemm_epilogue =
        pat.Op(fused_op_name_,
               {{
                   {"in_num_col_dims", pat.Attr("in_num_col_dims")},
                   {"activation_type", pat.Attr("activation_type")},
                   {"padding_weights", pat.Attr("padding_weights")},
               }});
    std::unordered_map<std::string, paddle::drr::Attribute> act_attrs;
    if (act_type_ == "gelu") {
      act_attrs.emplace("approximate", pat.Attr("approximate"));
    }
    const auto &act = pat.Op(activation_type[act_type_], act_attrs);

    gemm_epilogue({&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("y")},
                  {&pat.Tensor("gemm_epilogue_out")});
    act({&pat.Tensor("gemm_epilogue_out")}, {&pat.Tensor("act_out")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      const std::string &act_type =
          match_ctx.Attr<std::string>("activation_type");
      if (!act_type.empty()) return false;
      if (act_type_ == "gelu") {
        bool Attr_approx = match_ctx.Attr<bool>("approximate");
        if (!Attr_approx) return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    std::unordered_map<std::string, paddle::drr::Attribute> fused_attrs{
        {"in_num_col_dims", pat.Attr("in_num_col_dims")},
        {"activation_type", res.StrAttr(act_type_)},
        {"padding_weights", pat.Attr("padding_weights")},
    };
    const auto &gemm_epilogue_with_act = res.Op(fused_op_name_, fused_attrs);
    gemm_epilogue_with_act(
        {&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("y")},
        {&res.Tensor("act_out")});
  }
};

class MatmulAddActFusePass : public pir::PatternRewritePass {
 public:
  MatmulAddActFusePass()
      : pir::PatternRewritePass("matmul_add_act_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

    bool use_cutlass = false;
    if (Has(std::string("use_cutlass"))) {
      use_cutlass = Get<bool>(std::string("use_cutlass"));
    }
    if (use_cutlass) {
      /// MatmulAddPattern
      ps.Add(paddle::drr::Create<MatmulAddPattern>(
          context, paddle::dialect::GemmEpilogueOp::name(), true));
      ps.Add(paddle::drr::Create<MatmulAddPattern>(
          context, paddle::dialect::GemmEpilogueOp::name(), false));
      /// MatmulAddActPattern
      for (const auto &act_op : act_ops) {
        ps.Add(paddle::drr::Create<MatmulAddActPattern>(
            context, act_op, paddle::dialect::GemmEpilogueOp::name()));
      }
    }
    /// MatmulAddPatternw
    ps.Add(paddle::drr::Create<MatmulAddPattern>(
        context, paddle::dialect::FcOp::name(), false));
    /// MatmulAddActPattern
    ps.Add(paddle::drr::Create<MatmulAddActPattern>(
        context, "relu", paddle::dialect::FcOp::name()));

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateMatmulAddActFusePass() {
  return std::make_unique<MatmulAddActFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(matmul_add_act_fuse_pass, MatmulAddActFusePass);
