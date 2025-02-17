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

#include <optional>

#include "paddle/fluid/pir/transforms/general/delete_quant_dequant_linear_op_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/analysis_info.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class DeleteQuantDequantLinearOpBasePattern
    : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "DeleteQuantDequantLinearOpBasePattern";
  }

  explicit DeleteQuantDequantLinearOpBasePattern(
      paddle::framework::Scope* scope,
      std::reference_wrapper<std::optional<pir::detail::PassExecutionState>>
          pass_state)
      : scope_(scope), pass_state_(pass_state) {}

  bool Constraint(const paddle::drr::MatchContext& match_ctx) const {
    if (!pir::ValueIsPersistable(match_ctx.Tensor("scale"))) {
      return false;
    }
    auto input_scale_dtype =
        pir::GetDataTypeFromValue(match_ctx.Tensor("scale"));
    auto input_scale_name =
        pir::GetParameterNameFromValue(match_ctx.Tensor("scale"));
    auto* input_scale_var = this->scope_->FindVar(input_scale_name);
    PADDLE_ENFORCE_NOT_NULL(
        input_scale_var,
        common::errors::InvalidArgument("Persistable var [%s] not in scope.",
                                        input_scale_name));
    if (!input_scale_dtype.isa<pir::Float16Type>() &&
        !input_scale_dtype.isa<pir::Float32Type>()) {
      return false;
    }
    return true;
  }

  void PostProcess(const paddle::drr::MatchContext& match_ctx) const {
    float input_scale = 0;
    auto input_scale_dtype =
        pir::GetDataTypeFromValue(match_ctx.Tensor("scale"));
    auto input_scale_name =
        pir::GetParameterNameFromValue(match_ctx.Tensor("scale"));
    auto* input_scale_var = this->scope_->FindVar(input_scale_name);
    auto* input_scale_tensor = input_scale_var->GetMutable<phi::DenseTensor>();
    phi::DenseTensor temp_tensor;
    temp_tensor.Resize(input_scale_tensor->dims());
    paddle::framework::TensorCopySync(
        *input_scale_tensor, phi::CPUPlace{}, &temp_tensor);

    if (input_scale_dtype.isa<pir::Float16Type>()) {
      const phi::dtype::float16* input_scale_data =
          temp_tensor.data<phi::dtype::float16>();
      input_scale = static_cast<float>(input_scale_data[0]);
    } else {  // (input_scale_dtype.isa<pir::Float32Type>())
      const float* input_scale_data = temp_tensor.data<float>();
      input_scale = input_scale_data[0];
    }
    PADDLE_ENFORCE_EQ(
        this->pass_state_.get().has_value(),
        true,
        common::errors::InvalidArgument("pass state has no value"));

    auto& quant_analysis =
        this->pass_state_.get()->am.GetAnalysis<pir::pass::QuantAnalysis>();
    this->pass_state_.get()
        ->preserved_analyses.Preserve<pir::pass::QuantAnalysis>();
    PADDLE_ENFORCE_EQ(
        this->pass_state_.get()
            ->preserved_analyses.IsPreserved<pir::pass::QuantAnalysis>(),
        true,
        common::errors::InvalidArgument("QuantAnalysis should be Preserved"));
    quant_analysis.scale_map[match_ctx.Tensor("x")] =
        std::vector<float>({input_scale});

    // Save scale info into op that connected to dequantize_linear_op
    pir::Operation* op =
        match_ctx.Tensor("dequantize_linear_out").defining_op();
    auto next_op = pir::GetUseOpsForOutput(op, 0)[0].first;
    pir::IrContext* ctx = pir::IrContext::Instance();
    auto inputs = next_op->operands_source();
    std::vector<pir::Attribute> input_index_vec;
    std::vector<pir::Attribute> input_scale_vec;
    for (size_t i = 0; i < inputs.size(); i++) {
      auto input = inputs[i];
      if (match_ctx.Tensor("dequantize_linear_out") == input) {
        // non-weight quant-dequant situation
        pir::Attribute input_index =
            pir::Int32Attribute::get(ctx, static_cast<int>(i));
        input_index_vec.push_back(input_index);
        pir::Attribute input_scale_attr =
            pir::FloatAttribute::get(pir::IrContext::Instance(), input_scale);
        input_scale_vec.push_back(input_scale_attr);
      }
    }
    // set the scale info into op
    if (input_index_vec.size() > 0) {
      pir::Attribute inputs_index =
          pir::ArrayAttribute::get(pir::IrContext::Instance(), input_index_vec);
      next_op->set_attribute("inputs_index", inputs_index);
      pir::Attribute inputs_scale =
          pir::ArrayAttribute::get(pir::IrContext::Instance(), input_scale_vec);
      next_op->set_attribute("inputs_scale", inputs_scale);
    }
  }

  void operator()(paddle::drr::DrrPatternContext* ctx) const override {}

 protected:
  paddle::framework::Scope* scope_{nullptr};
  std::reference_wrapper<std::optional<pir::detail::PassExecutionState>>
      pass_state_;
};

class DeleteQuantDequantLinearOpWithNoneInputPattern
    : public DeleteQuantDequantLinearOpBasePattern {
 public:
  std::string name() const override {
    return "DeleteQuantDequantLinearOpWithNoneInputPattern";
  }

  explicit DeleteQuantDequantLinearOpWithNoneInputPattern(
      paddle::framework::Scope* scope,
      std::reference_wrapper<std::optional<pir::detail::PassExecutionState>>
          pass_state)
      : DeleteQuantDequantLinearOpBasePattern(scope, pass_state) {}

  void operator()(paddle::drr::DrrPatternContext* ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto& quantize_linear_op =
        pat.Op(paddle::dialect::QuantizeLinearOp::name());
    const auto& dequantize_linear_op =
        pat.Op(paddle::dialect::DequantizeLinearOp::name());
    quantize_linear_op({&pat.Tensor("x"),
                        &pat.Tensor("scale"),
                        &pat.Tensor("zero_point"),
                        &pat.InputNoneTensor(),
                        &pat.InputNoneTensor()},
                       {&pat.Tensor("quantize_linear_out"),
                        &pat.OutputNoneTensor(),
                        &pat.OutputNoneTensor(),
                        &pat.OutputNoneTensor()});
    dequantize_linear_op({&pat.Tensor("quantize_linear_out"),
                          &pat.Tensor("descale"),
                          &pat.Tensor("dezero_point"),
                          &pat.InputNoneTensor(),
                          &pat.InputNoneTensor()},
                         {&pat.Tensor("dequantize_linear_out"),
                          &pat.OutputNoneTensor(),
                          &pat.OutputNoneTensor(),
                          &pat.OutputNoneTensor()});

    pat.AddConstraint([this](const paddle::drr::MatchContext& match_ctx) {
      return this->Constraint(match_ctx);
    });

    pat.AddPostProcess([this](const paddle::drr::MatchContext& match_ctx) {
      this->PostProcess(match_ctx);
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    res.Tensor("dequantize_linear_out").Assign(res.Tensor("x"));
  }
};

class DeleteQuantDequantLinearOpPattern
    : public DeleteQuantDequantLinearOpBasePattern {
 public:
  std::string name() const override {
    return "DeleteQuantDequantLinearOpPattern";
  }

  explicit DeleteQuantDequantLinearOpPattern(
      paddle::framework::Scope* scope,
      std::reference_wrapper<std::optional<pir::detail::PassExecutionState>>
          pass_state)
      : DeleteQuantDequantLinearOpBasePattern(scope, pass_state) {}

  void operator()(paddle::drr::DrrPatternContext* ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto& quantize_linear_op =
        pat.Op(paddle::dialect::QuantizeLinearOp::name());
    const auto& dequantize_linear_op =
        pat.Op(paddle::dialect::DequantizeLinearOp::name());
    quantize_linear_op({&pat.Tensor("x"),
                        &pat.Tensor("scale"),
                        &pat.Tensor("zero_point"),
                        &pat.Tensor("in_accum"),
                        &pat.Tensor("in_state")},
                       {&pat.Tensor("quantize_linear_out"),
                        &pat.Tensor("state_out"),
                        &pat.Tensor("accum_out"),
                        &pat.Tensor("scale_out")});
    dequantize_linear_op({&pat.Tensor("quantize_linear_out"),
                          &pat.Tensor("descale"),
                          &pat.Tensor("dezero_point"),
                          &pat.Tensor("dein_accum"),
                          &pat.Tensor("dein_state")},
                         {&pat.Tensor("dequantize_linear_out"),
                          &pat.Tensor("destate_out"),
                          &pat.Tensor("deaccum_out"),
                          &pat.Tensor("descale_out")});

    pat.AddConstraint([this](const paddle::drr::MatchContext& match_ctx) {
      return this->Constraint(match_ctx);
    });
    pat.AddPostProcess([this](const paddle::drr::MatchContext& match_ctx) {
      this->PostProcess(match_ctx);
    });
    paddle::drr::ResultPattern res = pat.ResultPattern();
    res.Tensor("dequantize_linear_out").Assign(res.Tensor("x"));
  }
};

class DeleteQuantDequantLinearOpPass : public pir::PatternRewritePass {
 public:
  DeleteQuantDequantLinearOpPass()
      : pir::PatternRewritePass("delete_quant_dequant_linear_op_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    PADDLE_ENFORCE_EQ(Has(pir::Pass::kParamScopeAttr),
                      true,
                      common::errors::InvalidArgument(
                          "Pass initialize failed."
                          "When using DeleteQuantDequantLinearOpPass, scope "
                          "attribute is required!"
                          "Use Set method to set the scope attribute."));
    scope_ = &Get<paddle::framework::Scope>(pir::Pass::kParamScopeAttr);
    PADDLE_ENFORCE_NOT_NULL(
        scope_, common::errors::InvalidArgument("scope can not be nullptr"));
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<DeleteQuantDequantLinearOpPattern>(
        context, scope_, std::ref(pass_state())));
    ps.Add(paddle::drr::Create<DeleteQuantDequantLinearOpWithNoneInputPattern>(
        context, scope_, std::ref(pass_state())));

    return ps;
  }

 private:
  paddle::framework::Scope* scope_{nullptr};
};
}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateDeleteQuantDequantLinearOpPass() {
  return std::make_unique<DeleteQuantDequantLinearOpPass>();
}

}  // namespace pir

REGISTER_IR_PASS(delete_quant_dequant_linear_op_pass,
                 DeleteQuantDequantLinearOpPass);
