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

#include "paddle/fluid/pir/transforms/general/delete_weight_dequant_linear_op_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/analysis_info.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class DeleteWeightDequantLinearOpPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "DeleteWeightDequantLinearOpPattern";
  }

  explicit DeleteWeightDequantLinearOpPattern(
      paddle::framework::Scope* scope,
      std::reference_wrapper<std::optional<pir::detail::PassExecutionState>>
          pass_state)
      : scope_(scope), pass_state_(pass_state) {}

  void operator()(paddle::drr::DrrPatternContext* ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto& dequantize_linear_op =
        pat.Op(paddle::dialect::DequantizeLinearOp::name());

    dequantize_linear_op({&pat.Tensor("weight"),
                          &pat.Tensor("scale"),
                          &pat.Tensor("dezero_point"),
                          &pat.InputNoneTensor(),
                          &pat.InputNoneTensor()},
                         {&pat.Tensor("dequantize_linear_out"),
                          &pat.OutputNoneTensor(),
                          &pat.OutputNoneTensor(),
                          &pat.OutputNoneTensor()});

    pat.AddConstraint([this](const paddle::drr::MatchContext& match_ctx) {
      if (!pir::ValueIsPersistable(match_ctx.Tensor("weight"))) {
        return false;
      }
      if (!pir::ValueIsPersistable(match_ctx.Tensor("scale"))) {
        return false;
      }
      auto weight_scale_dtype =
          pir::GetDataTypeFromValue(match_ctx.Tensor("scale"));
      auto weight_scale_name =
          pir::GetParameterNameFromValue(match_ctx.Tensor("scale"));
      auto* weight_scale_var = this->scope_->FindVar(weight_scale_name);
      PADDLE_ENFORCE_NOT_NULL(
          weight_scale_var,
          common::errors::InvalidArgument("Persistable var [%s] not in scope.",
                                          weight_scale_name));
      if (!weight_scale_dtype.isa<pir::Float16Type>() &&
          !weight_scale_dtype.isa<pir::Float32Type>()) {
        return false;
      }
      return true;
    });

    pat.AddPostProcess([this](
                           const paddle::drr::MatchContext& match_ctx) mutable {
      pir::Operation* op =
          match_ctx.Tensor("dequantize_linear_out").defining_op();
      auto next_op_list = pir::GetUseOpsForOutput(op, 0);
      for (auto const& [next_op, op_index] : next_op_list) {
        if (next_op->isa<paddle::dialect::Conv2dOp>() ||
            next_op->isa<paddle::dialect::MatmulOp>() ||
            next_op->isa<paddle::dialect::DepthwiseConv2dOp>() ||
            next_op->isa<paddle::dialect::Conv2dTransposeOp>()) {
          std::vector<float> weight_scales;
          int quant_axis = op->attribute("quant_axis")
                               .dyn_cast<pir::Int32Attribute>()
                               .data();
          auto weight_scale_dtype =
              pir::GetDataTypeFromValue(match_ctx.Tensor("scale"));
          auto weight_scale_name =
              pir::GetParameterNameFromValue(match_ctx.Tensor("scale"));
          auto* weight_scale_var = this->scope_->FindVar(weight_scale_name);
          auto* weight_scale_tensor =
              weight_scale_var->GetMutable<phi::DenseTensor>();
          auto weight_scale_nums = weight_scale_tensor->numel();

          if (quant_axis == -1) {
            PADDLE_ENFORCE_EQ(
                weight_scale_nums,
                1,
                common::errors::InvalidArgument(
                    "When quant_axis == -1, it means using per_layer "
                    "dequantization. In this situation, the number of "
                    "weight_scale should be 1, but received %d.",
                    weight_scale_nums));
          } else {
            std::vector<int64_t> weight_shape =
                pir::GetShapeFromValue(match_ctx.Tensor("weight"));
            quant_axis =
                quant_axis >= 0 ? quant_axis : quant_axis + weight_shape.size();
            PADDLE_ENFORCE_EQ(
                weight_scale_nums,
                weight_shape[quant_axis],
                common::errors::InvalidArgument(
                    "When quant_axis != -1, it means using per_channel "
                    "dequantization. In this situation, the number of "
                    "weight_scale should be equal with "
                    "weight_shape[quant_axis=%d]=%ld , but received "
                    "%d.",
                    quant_axis,
                    weight_shape[quant_axis],
                    weight_scale_nums));
          }

          if (weight_scale_dtype.isa<pir::Float16Type>()) {
            const phi::dtype::float16* weight_scale_data =
                weight_scale_tensor->data<phi::dtype::float16>();
            for (int i = 0; i < weight_scale_nums; i++) {
              weight_scales.push_back(weight_scale_data[i]);
            }
          } else {
            const float* weight_scale_data = weight_scale_tensor->data<float>();
            for (int i = 0; i < weight_scale_nums; i++) {
              weight_scales.push_back(weight_scale_data[i]);
            }
          }

          PADDLE_ENFORCE_EQ(
              this->pass_state_.get().has_value(),
              true,
              common::errors::InvalidArgument("pass state has no value"));

          auto& quant_analysis =
              this->pass_state_.get()
                  ->am.GetAnalysis<pir::pass::QuantAnalysis>();
          this->pass_state_.get()
              ->preserved_analyses.Preserve<pir::pass::QuantAnalysis>();

          PADDLE_ENFORCE_EQ(
              this->pass_state_.get()
                  ->preserved_analyses.IsPreserved<pir::pass::QuantAnalysis>(),
              true,
              common::errors::InvalidArgument(
                  "QuantAnalysis should be Preserved"));
          quant_analysis.scale_map[match_ctx.Tensor("weight")] = weight_scales;

          auto& int8_analysis = this->pass_state_.get()
                                    ->am.GetAnalysis<pir::pass::Int8Analysis>();
          this->pass_state_.get()
              ->preserved_analyses.Preserve<pir::pass::Int8Analysis>();
          PADDLE_ENFORCE_EQ(
              this->pass_state_.get()
                  ->preserved_analyses.IsPreserved<pir::pass::Int8Analysis>(),
              true,
              common::errors::InvalidArgument(
                  "Int8Analysis should be Preserved"));
          int8_analysis.enable_int8 = true;
        }
      }
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    res.Tensor("dequantize_linear_out").Assign(res.Tensor("weight"));
  }

 private:
  paddle::framework::Scope* scope_{nullptr};
  std::reference_wrapper<std::optional<pir::detail::PassExecutionState>>
      pass_state_;
};

class DeleteWeightDequantLinearOpPass : public pir::PatternRewritePass {
 public:
  DeleteWeightDequantLinearOpPass()
      : pir::PatternRewritePass("delete_weight_dequant_linear_op_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    PADDLE_ENFORCE_EQ(Has(pir::Pass::kParamScopeAttr),
                      true,
                      common::errors::InvalidArgument(
                          "Pass initialize failed."
                          "When using DeleteWeightDequantLinearOpPass, scope "
                          "attribute is required!"
                          "Use Set method to set the scope attribute."));
    scope_ = &Get<paddle::framework::Scope>(pir::Pass::kParamScopeAttr);
    PADDLE_ENFORCE_NOT_NULL(
        scope_, common::errors::InvalidArgument("scope can not be nullptr"));
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<DeleteWeightDequantLinearOpPattern>(
        context, scope_, std::ref(pass_state())));
    return ps;
  }

 private:
  paddle::framework::Scope* scope_{nullptr};
};
}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateDeleteWeightDequantLinearOpPass() {
  return std::make_unique<DeleteWeightDequantLinearOpPass>();
}

}  // namespace pir

REGISTER_IR_PASS(delete_weight_dequant_linear_op_pass,
                 DeleteWeightDequantLinearOpPass);
