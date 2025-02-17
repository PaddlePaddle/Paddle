// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/tensorrt/trt_delete_weight_dequant_linear_op_pass.h"

#include <memory>

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/analysis_info.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/common/ddim.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class TrtDeleteWeightDequantLinearOpBasePattern
    : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "TrtDeleteWeightDequantLinearOpBasePattern";
  }

  explicit TrtDeleteWeightDequantLinearOpBasePattern(
      paddle::framework::Scope* scope,
      std::reference_wrapper<std::optional<pir::detail::PassExecutionState>>
          pass_state)
      : scope_(scope), pass_state_(pass_state) {}
  bool Constraint(const paddle::drr::MatchContext& match_ctx) const {
    if (!pir::ValueIsPersistable(match_ctx.Tensor("weight"))) {
      return false;
    }
    if (!pir::ValueIsPersistable(match_ctx.Tensor("scale"))) {
      return false;
    }
    pir::Operation* op =
        match_ctx.Tensor("dequantize_linear_out").defining_op();
    auto next_op_list = pir::GetUseOpsForOutput(op, 0);
    for (auto const& [next_op, op_index] : next_op_list) {
      if (!(next_op->isa<paddle::dialect::Conv2dOp>() ||
            next_op->isa<paddle::dialect::MatmulOp>() ||
            next_op->isa<paddle::dialect::DepthwiseConv2dOp>() ||
            next_op->isa<paddle::dialect::Conv2dTransposeOp>())) {
        return false;
      }
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
  }

  void PostProcess(const paddle::drr::MatchContext& match_ctx) const {
    pir::Operation* op =
        match_ctx.Tensor("dequantize_linear_out").defining_op();

    int bit_length =
        op->attribute("bit_length").dyn_cast<pir::Int32Attribute>().data();
    int range = ((1 << (bit_length - 1)) - 1);

    // get weight tensor
    auto weight_name =
        pir::GetParameterNameFromValue(match_ctx.Tensor("weight"));
    auto* weight_tensor =
        this->scope_->GetVar(weight_name)->GetMutable<phi::DenseTensor>();
    phi::DenseTensor weight_tensor_cpu;
    weight_tensor_cpu.Resize(weight_tensor->dims());
    paddle::framework::TensorCopySync(
        *weight_tensor, phi::CPUPlace{}, &weight_tensor_cpu);
    // int8 cast to float for calculation
    std::vector<float> quantized_weight_data;
    for (int i = 0; i < weight_tensor_cpu.numel(); i++) {
      if (weight_tensor_cpu.dtype() == phi::DataType::INT8) {
        quantized_weight_data.push_back(
            static_cast<float>(weight_tensor_cpu.data<int8_t>()[i]));
      } else {
        quantized_weight_data.push_back(weight_tensor_cpu.data<float>()[i]);
      }
    }
    auto w_dims = weight_tensor_cpu.dims();

    // Get weight scale
    std::vector<float> weight_scale;
    auto weight_scale_name =
        pir::GetParameterNameFromValue(match_ctx.Tensor("scale"));
    auto* weight_scale_tensor =
        this->scope_->GetVar(weight_scale_name)->GetMutable<phi::DenseTensor>();
    phi::DenseTensor weight_scale_tensor_tmp;
    weight_scale_tensor_tmp.Resize(weight_scale_tensor->dims());
    paddle::framework::TensorCopySync(
        *weight_scale_tensor, phi::CPUPlace{}, &weight_scale_tensor_tmp);
    weight_scale_tensor = &weight_scale_tensor_tmp;
    float* weight_scale_data = weight_scale_tensor->data<float>();
    auto weight_scale_nums = weight_scale_tensor->numel();
    weight_scale.reserve(weight_scale_nums);
    for (int i = 0; i < weight_scale_nums; i++) {
      weight_scale.push_back(weight_scale_data[i] / static_cast<float>(range));
    }

    // dequant weight
    std::vector<float> weight_data_tmp;
    weight_data_tmp.reserve(weight_tensor_cpu.numel());
    int quant_axis =
        op->attribute("quant_axis").dyn_cast<pir::Int32Attribute>().data();

    if (quant_axis == -1) {  // per_layer quant_dequant: all OP
      PADDLE_ENFORCE_EQ(weight_scale_nums,
                        1,
                        common::errors::InvalidArgument(
                            "When quant_axis == -1 means use per_layer "
                            "quant_dequant, weight_scale'number should be 1."));

      //  float(weight) * scale
      for (int i = 0; i < weight_tensor_cpu.numel(); i++) {
        weight_data_tmp.push_back(quantized_weight_data[i] * weight_scale[0]);
      }

    } else if (quant_axis == 0) {  // per_channel quant_dequant: conv2d,
                                   // depthwise_conv2d, fused_conv2d_add_act
      PADDLE_ENFORCE_EQ(
          weight_scale_nums,
          w_dims[quant_axis],
          common::errors::InvalidArgument(
              "When quant_axis == 0 means use per_channel quant_dequant, "
              "weight_scale'numbers should be equal channels."));
      PADDLE_ENFORCE_EQ(
          w_dims.size(),
          4,
          common::errors::InvalidArgument(
              "When quant_axis == 0 means use per_channel "
              "quant_dequant, (conv2d, depthwise_conv2d, "
              "fused_conv2d_add_act)'s weight dims should be 4."));

      for (int i = 0; i < weight_tensor_cpu.numel(); i++) {
        int inner_size = static_cast<int>(w_dims[1] * w_dims[2] * w_dims[3]);
        weight_data_tmp.push_back(quantized_weight_data[i] *
                                  weight_scale[i / inner_size]);
      }
    } else if (quant_axis == 1) {
      PADDLE_ENFORCE_EQ(
          weight_scale_nums,
          w_dims[quant_axis],
          common::errors::InvalidArgument(
              "When quant_axis == 1 means use per_channel quant_dequant, "
              "weight_scale'numbers should be equal channels."));

      if (w_dims.size() == 4) {  // conv2d_transpose
        auto next_op_list = pir::GetUseOpsForOutput(op, 0);
        for (auto const& [next_op, op_index] : next_op_list) {
          PADDLE_ENFORCE_EQ(
              next_op->isa<paddle::dialect::Conv2dTransposeOp>(),
              true,
              common::errors::InvalidArgument(
                  "When quant_axis == 1 means use per_channel quant_dequant, "
                  "only conv2d_transpose weight dims equal 4."));
        }

        for (int i = 0; i < weight_tensor_cpu.numel(); i++) {
          int inner_size = static_cast<int>(w_dims[2] * w_dims[3]);
          weight_data_tmp.push_back(quantized_weight_data[i] *
                                    weight_scale[(i / inner_size) % w_dims[1]]);
        }
      } else if (w_dims.size() == 2) {
        for (int i = 0; i < weight_tensor_cpu.numel(); i++) {
          weight_data_tmp.push_back(quantized_weight_data[i] *
                                    weight_scale[i % w_dims[1]]);
        }
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "When quant_axis == 1 , weight dims should be 2 or 4, please check "
            "your model "));
      }
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "quant_axis should be -1 or 0 or 1, please check your model "
          "OP'attribute "));
    }

    // set dequant weight data into weight tensor
    auto weight_tensor_place = weight_tensor->place();
    weight_tensor->clear();  // clear int weight
    if (weight_tensor_place == phi::GPUPlace()) {
      auto* dev_ctx = static_cast<phi::GPUContext*>(
          phi::DeviceContextPool::Instance().Get(phi::GPUPlace()));
      paddle::framework::TensorFromVector(
          weight_data_tmp, *dev_ctx, weight_tensor);
    } else if (weight_tensor_place == phi::CPUPlace()) {
      auto* dev_ctx = static_cast<phi::CPUContext*>(
          phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
      paddle::framework::TensorFromVector(
          weight_data_tmp, *dev_ctx, weight_tensor);
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "trt_delete_weight_dequant_linear_op_pass only support cpu and gpu "
          "place now."));
    }
    weight_tensor->Resize(common::make_ddim(common::vectorize(w_dims)));
  }

  void operator()(paddle::drr::DrrPatternContext* ctx) const override {}

 private:
  paddle::framework::Scope* scope_{nullptr};
  std::reference_wrapper<std::optional<pir::detail::PassExecutionState>>
      pass_state_;
};

class TrtDeleteWeightDequantLinearOpWithNoneInputPattern
    : public TrtDeleteWeightDequantLinearOpBasePattern {
 public:
  std::string name() const override {
    return "TrtDeleteWeightDequantLinearOpWithNoneInputPattern";
  }

  explicit TrtDeleteWeightDequantLinearOpWithNoneInputPattern(
      paddle::framework::Scope* scope,
      std::reference_wrapper<std::optional<pir::detail::PassExecutionState>>
          pass_state)
      : TrtDeleteWeightDequantLinearOpBasePattern(scope, pass_state) {}

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
      return this->Constraint(match_ctx);
    });

    pat.AddPostProcess(
        [this](const paddle::drr::MatchContext& match_ctx) mutable {
          this->PostProcess(match_ctx);
        });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    res.Tensor("dequantize_linear_out").Assign(res.Tensor("weight"));
  }
};

class TrtDeleteWeightDequantLinearOpPattern
    : public TrtDeleteWeightDequantLinearOpBasePattern {
 public:
  std::string name() const override {
    return "TrtDeleteWeightDequantLinearOpPattern";
  }

  explicit TrtDeleteWeightDequantLinearOpPattern(
      paddle::framework::Scope* scope,
      std::reference_wrapper<std::optional<pir::detail::PassExecutionState>>
          pass_state)
      : TrtDeleteWeightDequantLinearOpBasePattern(scope, pass_state) {}

  void operator()(paddle::drr::DrrPatternContext* ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto& dequantize_linear_op =
        pat.Op(paddle::dialect::DequantizeLinearOp::name());

    dequantize_linear_op({&pat.Tensor("weight"),
                          &pat.Tensor("scale"),
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

    pat.AddPostProcess(
        [this](const paddle::drr::MatchContext& match_ctx) mutable {
          this->PostProcess(match_ctx);
        });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    res.Tensor("dequantize_linear_out").Assign(res.Tensor("weight"));
  }
};

class TrtDeleteWeightDequantLinearOpPatternPass
    : public pir::PatternRewritePass {
 public:
  TrtDeleteWeightDequantLinearOpPatternPass()
      : pir::PatternRewritePass("trt_delete_weight_dequant_linear_op_pass", 1) {
  }

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    PADDLE_ENFORCE_EQ(
        Has(pir::Pass::kParamScopeAttr),
        true,
        common::errors::InvalidArgument(
            "Pass initialize failed."
            "When using TrtDeleteWeightDequantLinearOpPatternPass, scope "
            "attribute is required!"
            "Use Set method to set the scope attribute."));
    scope_ = &Get<paddle::framework::Scope>(pir::Pass::kParamScopeAttr);
    PADDLE_ENFORCE_NOT_NULL(
        scope_, common::errors::InvalidArgument("scope can not be nullptr"));
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<TrtDeleteWeightDequantLinearOpPattern>(
        context, scope_, std::ref(pass_state())));
    ps.Add(
        paddle::drr::Create<TrtDeleteWeightDequantLinearOpWithNoneInputPattern>(
            context, scope_, std::ref(pass_state())));
    return ps;
  }

 private:
  paddle::framework::Scope* scope_{nullptr};
};
}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateTrtDeleteWeightDequantLinearOpPatternPass() {
  return std::make_unique<TrtDeleteWeightDequantLinearOpPatternPass>();
}

}  // namespace pir

REGISTER_IR_PASS(trt_delete_weight_dequant_linear_op_pass,
                 TrtDeleteWeightDequantLinearOpPatternPass);
