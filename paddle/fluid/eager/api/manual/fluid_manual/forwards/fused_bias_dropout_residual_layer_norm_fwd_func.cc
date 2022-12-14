// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/amp_auto_cast.h"
#include "paddle/fluid/eager/amp_utils.h"
#include "paddle/fluid/eager/api/manual/fluid_manual/dygraph_forward_api.h"
#include "paddle/fluid/eager/api/manual/fluid_manual/nodes/nodes.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
fused_bias_dropout_residual_layer_norm_dygraph_function(
    const paddle::experimental::Tensor& X,
    const paddle::experimental::Tensor& Residual,
    const paddle::experimental::Tensor& Bias,
    const paddle::experimental::Tensor& LnScale,
    const paddle::experimental::Tensor& LnBias,
    const paddle::framework::AttributeMap& attr_map) {
  paddle::platform::RecordEvent dygraph_entrance_record_event(
      "fused_bias_dropout_residual_layer_norm dygraph",
      paddle::platform::TracerEventType::Operator,
      1);
  VLOG(3) << "Running Eager Forward Op: fused_bias_dropout_residual_layer_norm";
  // Dygraph Forward Pass

  if (egr::Controller::Instance().GetAMPLevel() !=
      paddle::imperative::AmpLevel::O0) {
    VLOG(5) << "Check and Prepare For AMP";

    paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                         egr::kSlotSmallVectorSize>
        amp_tensors_vector = {{X}, {Residual}};
    if (Bias.initialized()) amp_tensors_vector.push_back({Bias});
    if (LnScale.initialized()) amp_tensors_vector.push_back({LnScale});
    if (LnBias.initialized()) amp_tensors_vector.push_back({LnBias});

    auto amp_dst_dtype = egr::GetAmpDestDtype(
        "fused_bias_dropout_residual_layer_norm", amp_tensors_vector);

    auto NEW_X = egr::AmpAutoCast(
        "X", X, amp_dst_dtype, "fused_bias_dropout_residual_layer_norm");
    auto NEW_Residual =
        egr::AmpAutoCast("Residual",
                         Residual,
                         amp_dst_dtype,
                         "fused_bias_dropout_residual_layer_norm");
    auto NEW_Bias =
        ((Bias.initialized())
             ? egr::AmpAutoCast("Bias",
                                Bias,
                                amp_dst_dtype,
                                "fused_bias_dropout_residual_layer_norm")
             : Bias);
    auto NEW_LnScale =
        ((LnScale.initialized())
             ? egr::AmpAutoCast("LnScale",
                                LnScale,
                                amp_dst_dtype,
                                "fused_bias_dropout_residual_layer_norm")
             : LnScale);
    auto NEW_LnBias =
        ((LnBias.initialized())
             ? egr::AmpAutoCast("LnBias",
                                LnBias,
                                amp_dst_dtype,
                                "fused_bias_dropout_residual_layer_norm")
             : LnBias);

    {
      paddle::imperative::AutoCastGuard guard(
          egr::Controller::Instance().GetCurrentTracer(),
          paddle::imperative::AmpLevel::O0);
      return fused_bias_dropout_residual_layer_norm_dygraph_function(
          NEW_X, NEW_Residual, NEW_Bias, NEW_LnScale, NEW_LnBias, attr_map);
    }
  }

  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> ins =
      {{"X", egr::EagerUtils::TrySyncToVars(X)},
       {"Residual", egr::EagerUtils::TrySyncToVars(Residual)}};
  if (Bias.initialized()) ins["Bias"] = egr::EagerUtils::TrySyncToVars(Bias);
  if (LnScale.initialized())
    ins["LnScale"] = egr::EagerUtils::TrySyncToVars(LnScale);
  if (LnBias.initialized())
    ins["LnBias"] = egr::EagerUtils::TrySyncToVars(LnBias);

  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> outs =
      {{"BiasDropoutResidualOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"DropoutMaskOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"LnMean",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"LnVariance",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"Y",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}}};

  // Prepare Autograd Meta
  egr::AutogradMeta* p_autograd_X = egr::EagerUtils::nullable_autograd_meta(X);
  egr::AutogradMeta* p_autograd_Residual =
      egr::EagerUtils::nullable_autograd_meta(Residual);
  egr::AutogradMeta* p_autograd_Bias =
      egr::EagerUtils::nullable_autograd_meta(Bias);
  egr::AutogradMeta* p_autograd_LnScale =
      egr::EagerUtils::nullable_autograd_meta(LnScale);
  egr::AutogradMeta* p_autograd_LnBias =
      egr::EagerUtils::nullable_autograd_meta(LnBias);

  bool trace_backward = egr::Controller::Instance().HasGrad();

  bool require_any_grad =
      egr::EagerUtils::ComputeRequireGrad(trace_backward,
                                          p_autograd_X,
                                          p_autograd_Residual,
                                          p_autograd_Bias,
                                          p_autograd_LnScale,
                                          p_autograd_LnBias);

  paddle::framework::AttributeMap attrs = attr_map;
  paddle::framework::AttributeMap default_attrs;
  egr::Controller::Instance().GetCurrentTracer()->TraceOp(
      "fused_bias_dropout_residual_layer_norm",
      ins,
      outs,
      attrs,
      egr::Controller::Instance().GetExpectedPlace(),
      &default_attrs,
      true,
      {});

  paddle::experimental::Tensor BiasDropoutResidualOut;
  egr::EagerUtils::GetOutput(outs["BiasDropoutResidualOut"][0],
                             &BiasDropoutResidualOut);
  paddle::experimental::Tensor DropoutMaskOut;
  egr::EagerUtils::GetOutput(outs["DropoutMaskOut"][0], &DropoutMaskOut);
  paddle::experimental::Tensor LnMean;
  egr::EagerUtils::GetOutput(outs["LnMean"][0], &LnMean);
  paddle::experimental::Tensor LnVariance;
  egr::EagerUtils::GetOutput(outs["LnVariance"][0], &LnVariance);
  paddle::experimental::Tensor Y;
  egr::EagerUtils::GetOutput(outs["Y"][0], &Y);

  {
    paddle::platform::RecordEvent node_creation_record_event(
        "fused_bias_dropout_residual_layer_norm node_creation",
        paddle::platform::TracerEventType::OperatorInner,
        1);
    egr::AutogradMeta* p_autograd_BiasDropoutResidualOut =
        egr::EagerUtils::autograd_meta(&BiasDropoutResidualOut);
    egr::AutogradMeta* p_autograd_DropoutMaskOut =
        egr::EagerUtils::autograd_meta(&DropoutMaskOut);
    egr::AutogradMeta* p_autograd_LnMean =
        egr::EagerUtils::autograd_meta(&LnMean);
    egr::AutogradMeta* p_autograd_LnVariance =
        egr::EagerUtils::autograd_meta(&LnVariance);
    egr::AutogradMeta* p_autograd_Y = egr::EagerUtils::autograd_meta(&Y);
    if (require_any_grad) {
      VLOG(6) << " Construct Grad for fused_bias_dropout_residual_layer_norm ";
      egr::EagerUtils::PassStopGradient(false,
                                        p_autograd_BiasDropoutResidualOut,
                                        p_autograd_DropoutMaskOut,
                                        p_autograd_LnMean,
                                        p_autograd_LnVariance,
                                        p_autograd_Y);
      // Create GradOpNode
      auto grad_node =
          std::shared_ptr<fused_bias_dropout_residual_layer_normGradNodeCompat>(
              new fused_bias_dropout_residual_layer_normGradNodeCompat(5, 5));

      // Set Attributes
      grad_node->SetAttrMap(std::move(attrs));
      grad_node->SetDefaultAttrMap(std::move(default_attrs));

      // Set Tensor Wrappers
      grad_node->SetTensorWrapperBias(Bias);
      grad_node->SetTensorWrapperBiasDropoutResidualOut(BiasDropoutResidualOut);
      grad_node->SetTensorWrapperDropoutMaskOut(DropoutMaskOut);
      grad_node->SetTensorWrapperLnBias(LnBias);
      grad_node->SetTensorWrapperLnMean(LnMean);
      grad_node->SetTensorWrapperLnScale(LnScale);
      grad_node->SetTensorWrapperLnVariance(LnVariance);
      grad_node->SetTensorWrapperResidual(Residual);
      grad_node->SetTensorWrapperX(X);

      grad_node->SetGradOutMeta(X, 0);
      grad_node->SetGradOutMeta(Residual, 1);
      grad_node->SetGradOutMeta(Bias, 2);
      grad_node->SetGradOutMeta(LnScale, 3);
      grad_node->SetGradOutMeta(LnBias, 4);

      egr::EagerUtils::SetOutRankWithSlot(p_autograd_BiasDropoutResidualOut, 0);
      grad_node->SetGradInMeta(BiasDropoutResidualOut, 0);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_DropoutMaskOut, 1);
      grad_node->SetGradInMeta(DropoutMaskOut, 1);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_LnMean, 2);
      grad_node->SetGradInMeta(LnMean, 2);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_LnVariance, 3);
      grad_node->SetGradInMeta(LnVariance, 3);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_Y, 4);
      egr::EagerUtils::SetHistory(p_autograd_Y, grad_node);
      grad_node->SetGradInMeta(Y, 4);
      egr::EagerUtils::CheckAndRetainGrad(Y);
    }
  }

  return std::make_tuple(
      BiasDropoutResidualOut, DropoutMaskOut, LnMean, LnVariance, Y);
}
