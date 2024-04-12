// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/gpu/fc_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include <iostream>

namespace {

// 当前pass有两种Op实现，FcOp(cublasLt) 和 GemmEpilogueOp(cutlass)
// 用户可以通过python API `exp_enable_use_cutlass()` 和 C++ API `Exp_EnableUseCutlass()` 来选择是否启用cutlass实现的Op
// 如果不开启cutlass 那么使用FcOp，只能匹配[M, N]+[1,N]的模式，激活只支持["","relu"]
// 如果开启cutlass 那么使用GemmEpilogueOp, 额外支持[M, N]+[M, N]的模式，激活支持["", "relu", "gelu"]
// 需要注意的是：两种Op共用FCInferMeta函数，我们放宽了该函数的约束以匹配额外模式。也就是说FcOp不能处理的模式，目前只在pass的约束中过滤，FCInferMeta中的check被取消了。
std::set<std::string> act_ops = {{paddle::dialect::GeluOp::name()},
                                 {paddle::dialect::ReluOp::name()},};
std::unordered_map<std::string, std::string> activation_type = {
    {paddle::dialect::GeluOp::name(), "gelu"},
    {paddle::dialect::ReluOp::name(), "relu"},};                                 

class MatmulAddPattern : public paddle::drr::DrrPatternBase {
 private:
 std::string fused_op_name_;
 bool reverse_add_;

 public:
  explicit MatmulAddPattern(std::string fused_op_name, bool reverse_add)
      : fused_op_name_(fused_op_name), reverse_add_(reverse_add) {}
 
  std::string name() const override { return "MatmulAddPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("transpose_x")},
                                 {"transpose_y", pat.Attr("transpose_y")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    matmul({&pat.Tensor("x"), &pat.Tensor("w")}, {&pat.Tensor("matmul_out")});
    pat.Tensor("add_out") = reverse_add_ ? add(pat.Tensor("y"), pat.Tensor("matmul_out"))
                                         : add(pat.Tensor("matmul_out"), pat.Tensor("y"));

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      auto w_dims = pir::GetShapeFromValue(match_ctx.Tensor("w"));
      auto x_dims = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      auto y_dims = pir::GetShapeFromValue(match_ctx.Tensor("y"));
      if (w_dims.size() != 2 || x_dims.size() < 2) {
        return false;
      }
      if (x_dims.at(x_dims.size() - 1) != w_dims.at(0) ||
          match_ctx.Attr<bool>("transpose_x") == true ||
          match_ctx.Attr<bool>("transpose_y") == true) {
        return false;
      }
      
      if (y_dims.size() == 1) {
        return y_dims.at(0) == w_dims.at(1);
      }

      if(fused_op_name_ == paddle::dialect::FcOp::name()){
        if (y_dims.size() == 2) {
          return y_dims.at(0) == 1 && y_dims.at(1) == w_dims.at(1);
        }
      }
      else{
        // kai mod 要融合 M*N + N 和 M*N + M*N 两种elementwiseAdd模式。
        // 要求bias的维度：如果是1，那么需要是[N]。如果是2，那么要么是[1,N],要么是[M,N]。如果是大于2，则除最后一维外，和x相同；最后一维和w_dims.at[1]相同。
        if (y_dims.size() == x_dims.size()){
          if (y_dims.size() == 2) {
            return ((y_dims.at(0) == 1) || (y_dims.at(0) == x_dims.at(0))) && y_dims.at(1) == w_dims.at(1);
          }
          for(size_t ii = 0; ii < x_dims.size()-1; ii++){
            if(y_dims.at(ii) != x_dims.at(ii)){
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
    const auto &fc = res.Op(fused_op_name_,
                            {{
                                {"in_num_col_dims", in_num_col_dims_attr},
                                {"activation_type", res.StrAttr("")},
                                {"padding_weights", res.BoolAttr(false)},
                            }});
    fc({&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("y")},
       {&res.Tensor("add_out")});
  }
};

/// 当前只支持[Relu, Gelu]
class FcWithActPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string act_type_;
  std::string fused_op_name_;

 public:
  explicit FcWithActPattern(std::string act_type, std::string fused_op_name)
      : act_type_(act_type), fused_op_name_(fused_op_name) {}

  std::string name() const override { return "FcWithActPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &fc = pat.Op(fused_op_name_,
                      {{
                          {"in_num_col_dims", pat.Attr("in_num_col_dims")},
                          {"activation_type", pat.Attr("activation_type")},
                          {"padding_weights", pat.Attr("padding_weights")},
                      }});
    std::unordered_map<std::string, paddle::drr::Attribute> act_attrs;
    if (act_type_ == paddle::dialect::GeluOp::name()) {
      act_attrs.emplace("approximate", pat.Attr("approximate"));
    }
    const auto &act = pat.Op(act_type_, act_attrs);

    fc({&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("y")},
       {&pat.Tensor("fc_out")});
    act({&pat.Tensor("fc_out")}, {&pat.Tensor("act_out")});

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
        bool isEmpty_act = match_ctx.Attr<std::string>("activation_type").empty();
        if(!isEmpty_act) return false;
        if (act_type_ == paddle::dialect::GeluOp::name()) {
          bool Attr_approx = match_ctx.Attr<bool>("approximate");
          // 参考onednn实现。这里的意思我理解是，不支持gelu的估算。cutlass也没有approx参数。
          if (Attr_approx) return false;    
        }  
        return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    std::unordered_map<std::string, paddle::drr::Attribute> fused_attrs{
      {"in_num_col_dims", pat.Attr("in_num_col_dims")},
      {"activation_type", res.StrAttr(activation_type[act_type_])},
      {"padding_weights", pat.Attr("padding_weights")},
    };
    const auto &fc_with_act = res.Op(fused_op_name_, fused_attrs);
    fc_with_act({&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("y")},
                 {&res.Tensor("act_out")});
  }
};

class FcFusePass : public pir::PatternRewritePass {
 public:
  FcFusePass() : pir::PatternRewritePass("fc_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

    // Set(std::string("use_cutlass"), new bool(true));
    bool use_cutlass = false;
    if(Has(std::string("use_cutlass"))){
      use_cutlass = Get<bool>(std::string("use_cutlass"));
    }
    // if(use_cutlass){
    //   printf("use_cutlass !\n");
    // }

    if(use_cutlass){
      /// MatmulAddPattern
      ps.Add(paddle::drr::Create<MatmulAddPattern>(context, paddle::dialect::GemmEpilogueOp::name(), true));
      ps.Add(paddle::drr::Create<MatmulAddPattern>(context, paddle::dialect::GemmEpilogueOp::name(), false));
      /// FcWithActPattern
      for(const auto& act_op: act_ops){
        ps.Add(paddle::drr::Create<FcWithActPattern>(context, act_op, paddle::dialect::GemmEpilogueOp::name()));
      }
    }
    else{
      /// MatmulAddPattern
      ps.Add(paddle::drr::Create<MatmulAddPattern>(context, paddle::dialect::FcOp::name(), false));
      /// FcWithActPattern
      ps.Add(paddle::drr::Create<FcWithActPattern>(context, paddle::dialect::ReluOp::name(), paddle::dialect::FcOp::name()));
    }
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFcFusePass() {
  return std::make_unique<FcFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(fc_fuse_pass, FcFusePass);
