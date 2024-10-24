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

#include "paddle/fluid/eager/amp_auto_cast.h"
#include "paddle/fluid/eager/api/manual/fluid_manual/dygraph_forward_api.h"
#include "paddle/fluid/eager/api/manual/fluid_manual/nodes/nodes.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/imperative/amp_utils.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

std::tuple<paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor>
fused_feedforward_dygraph_function(
    const paddle::Tensor& X,
    const paddle::Tensor& Dropout1Seed,
    const paddle::Tensor& Dropout2Seed,
    const paddle::Tensor& Linear1Weight,
    const paddle::Tensor& Linear1Bias,
    const paddle::Tensor& Linear2Weight,
    const paddle::Tensor& Linear2Bias,
    const paddle::Tensor& Ln1Scale,
    const paddle::Tensor& Ln1Bias,
    const paddle::Tensor& Ln2Scale,
    const paddle::Tensor& Ln2Bias,
    const paddle::framework::AttributeMap& attr_map) {
  phi::RecordEvent dygraph_entrance_record_event(
      "fused_feedforward dygraph", phi::TracerEventType::Operator, 1);
  VLOG(3) << "Running Eager Forward Op: fused_feedforward";
  // Dygraph Forward Pass

  if (egr::Controller::Instance().GetAMPLevel() !=
      paddle::imperative::AmpLevel::O0) {
    VLOG(5) << "Check and Prepare For AMP";

    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        amp_tensors_vector = {{X}, {Linear1Weight}, {Linear2Weight}};
    if (Dropout1Seed.initialized())
      amp_tensors_vector.push_back({Dropout1Seed});
    if (Dropout2Seed.initialized())
      amp_tensors_vector.push_back({Dropout2Seed});
    if (Linear1Bias.initialized()) amp_tensors_vector.push_back({Linear1Bias});
    if (Linear2Bias.initialized()) amp_tensors_vector.push_back({Linear2Bias});
    if (Ln1Scale.initialized()) amp_tensors_vector.push_back({Ln1Scale});
    if (Ln1Bias.initialized()) amp_tensors_vector.push_back({Ln1Bias});
    if (Ln2Scale.initialized()) amp_tensors_vector.push_back({Ln2Scale});
    if (Ln2Bias.initialized()) amp_tensors_vector.push_back({Ln2Bias});

    auto amp_dst_dtype = paddle::imperative::GetAmpDestDtype(
        "fused_feedforward", amp_tensors_vector);

    auto NEW_X = egr::AmpAutoCast("X", X, amp_dst_dtype, "fused_feedforward");
    auto NEW_Linear1Weight = egr::AmpAutoCast(
        "Linear1Weight", Linear1Weight, amp_dst_dtype, "fused_feedforward");
    auto NEW_Linear2Weight = egr::AmpAutoCast(
        "Linear2Weight", Linear2Weight, amp_dst_dtype, "fused_feedforward");
    auto NEW_Dropout1Seed =
        ((Dropout1Seed.initialized()) ? egr::AmpAutoCast("Dropout1Seed",
                                                         Dropout1Seed,
                                                         amp_dst_dtype,
                                                         "fused_feedforward")
                                      : Dropout1Seed);
    auto NEW_Dropout2Seed =
        ((Dropout2Seed.initialized()) ? egr::AmpAutoCast("Dropout2Seed",
                                                         Dropout2Seed,
                                                         amp_dst_dtype,
                                                         "fused_feedforward")
                                      : Dropout2Seed);
    auto NEW_Linear1Bias =
        ((Linear1Bias.initialized()) ? egr::AmpAutoCast("Linear1Bias",
                                                        Linear1Bias,
                                                        amp_dst_dtype,
                                                        "fused_feedforward")
                                     : Linear1Bias);
    auto NEW_Linear2Bias =
        ((Linear2Bias.initialized()) ? egr::AmpAutoCast("Linear2Bias",
                                                        Linear2Bias,
                                                        amp_dst_dtype,
                                                        "fused_feedforward")
                                     : Linear2Bias);
    auto NEW_Ln1Scale =
        ((Ln1Scale.initialized())
             ? egr::AmpAutoCast(
                   "Ln1Scale", Ln1Scale, amp_dst_dtype, "fused_feedforward")
             : Ln1Scale);
    auto NEW_Ln1Bias =
        ((Ln1Bias.initialized())
             ? egr::AmpAutoCast(
                   "Ln1Bias", Ln1Bias, amp_dst_dtype, "fused_feedforward")
             : Ln1Bias);
    auto NEW_Ln2Scale =
        ((Ln2Scale.initialized())
             ? egr::AmpAutoCast(
                   "Ln2Scale", Ln2Scale, amp_dst_dtype, "fused_feedforward")
             : Ln2Scale);
    auto NEW_Ln2Bias =
        ((Ln2Bias.initialized())
             ? egr::AmpAutoCast(
                   "Ln2Bias", Ln2Bias, amp_dst_dtype, "fused_feedforward")
             : Ln2Bias);

    {
      paddle::imperative::AutoCastGuard guard(
          egr::Controller::Instance().GetCurrentAmpAttrs(),
          paddle::imperative::AmpLevel::O0);
      return fused_feedforward_dygraph_function(NEW_X,
                                                NEW_Dropout1Seed,
                                                NEW_Dropout2Seed,
                                                NEW_Linear1Weight,
                                                NEW_Linear1Bias,
                                                NEW_Linear2Weight,
                                                NEW_Linear2Bias,
                                                NEW_Ln1Scale,
                                                NEW_Ln1Bias,
                                                NEW_Ln2Scale,
                                                NEW_Ln2Bias,
                                                attr_map);
    }
  }

  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> ins =
      {{"X", egr::EagerUtils::TrySyncToVars(X)},
       {"Linear1Weight", egr::EagerUtils::TrySyncToVars(Linear1Weight)},
       {"Linear2Weight", egr::EagerUtils::TrySyncToVars(Linear2Weight)}};
  if (Dropout1Seed.initialized())
    ins["Dropout1Seed"] = egr::EagerUtils::TrySyncToVars(Dropout1Seed);
  if (Dropout2Seed.initialized())
    ins["Dropout2Seed"] = egr::EagerUtils::TrySyncToVars(Dropout2Seed);
  if (Linear1Bias.initialized())
    ins["Linear1Bias"] = egr::EagerUtils::TrySyncToVars(Linear1Bias);
  if (Linear2Bias.initialized())
    ins["Linear2Bias"] = egr::EagerUtils::TrySyncToVars(Linear2Bias);
  if (Ln1Scale.initialized())
    ins["Ln1Scale"] = egr::EagerUtils::TrySyncToVars(Ln1Scale);
  if (Ln1Bias.initialized())
    ins["Ln1Bias"] = egr::EagerUtils::TrySyncToVars(Ln1Bias);
  if (Ln2Scale.initialized())
    ins["Ln2Scale"] = egr::EagerUtils::TrySyncToVars(Ln2Scale);
  if (Ln2Bias.initialized())
    ins["Ln2Bias"] = egr::EagerUtils::TrySyncToVars(Ln2Bias);

  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> outs =
      {{"Out",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"Dropout1Mask",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"Dropout2Mask",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"Ln1Mean",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"Ln1Variance",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"Ln2Mean",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"Ln2Variance",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"Linear1Out",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"Ln1Out",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"Dropout1Out",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"Dropout2Out",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}}};

  // Prepare Autograd Meta
  egr::AutogradMeta* p_autograd_X = egr::EagerUtils::nullable_autograd_meta(X);
  egr::AutogradMeta* p_autograd_Dropout1Seed =
      egr::EagerUtils::nullable_autograd_meta(Dropout1Seed);
  egr::AutogradMeta* p_autograd_Dropout2Seed =
      egr::EagerUtils::nullable_autograd_meta(Dropout2Seed);
  egr::AutogradMeta* p_autograd_Linear1Weight =
      egr::EagerUtils::nullable_autograd_meta(Linear1Weight);
  egr::AutogradMeta* p_autograd_Linear1Bias =
      egr::EagerUtils::nullable_autograd_meta(Linear1Bias);
  egr::AutogradMeta* p_autograd_Linear2Weight =
      egr::EagerUtils::nullable_autograd_meta(Linear2Weight);
  egr::AutogradMeta* p_autograd_Linear2Bias =
      egr::EagerUtils::nullable_autograd_meta(Linear2Bias);
  egr::AutogradMeta* p_autograd_Ln1Scale =
      egr::EagerUtils::nullable_autograd_meta(Ln1Scale);
  egr::AutogradMeta* p_autograd_Ln1Bias =
      egr::EagerUtils::nullable_autograd_meta(Ln1Bias);
  egr::AutogradMeta* p_autograd_Ln2Scale =
      egr::EagerUtils::nullable_autograd_meta(Ln2Scale);
  egr::AutogradMeta* p_autograd_Ln2Bias =
      egr::EagerUtils::nullable_autograd_meta(Ln2Bias);

  bool trace_backward = egr::Controller::Instance().HasGrad();

  bool require_any_grad =
      egr::EagerUtils::ComputeRequireGrad(trace_backward,
                                          p_autograd_X,
                                          p_autograd_Dropout1Seed,
                                          p_autograd_Dropout2Seed,
                                          p_autograd_Linear1Weight,
                                          p_autograd_Linear1Bias,
                                          p_autograd_Linear2Weight,
                                          p_autograd_Linear2Bias,
                                          p_autograd_Ln1Scale,
                                          p_autograd_Ln1Bias,
                                          p_autograd_Ln2Scale,
                                          p_autograd_Ln2Bias);

  paddle::framework::AttributeMap attrs = attr_map;
  paddle::framework::AttributeMap default_attrs;
  egr::Controller::Instance().GetCurrentTracer()->TraceOp(
      "fused_feedforward",
      ins,
      outs,
      attrs,
      egr::Controller::Instance().GetExpectedPlace(),
      &default_attrs,
      true,
      {});

  paddle::Tensor Out;
  egr::EagerUtils::GetOutput(outs["Out"][0], &Out);
  paddle::Tensor Dropout1Mask;
  egr::EagerUtils::GetOutput(outs["Dropout1Mask"][0], &Dropout1Mask);
  paddle::Tensor Dropout2Mask;
  egr::EagerUtils::GetOutput(outs["Dropout2Mask"][0], &Dropout2Mask);
  paddle::Tensor Ln1Mean;
  egr::EagerUtils::GetOutput(outs["Ln1Mean"][0], &Ln1Mean);
  paddle::Tensor Ln1Variance;
  egr::EagerUtils::GetOutput(outs["Ln1Variance"][0], &Ln1Variance);
  paddle::Tensor Ln2Mean;
  egr::EagerUtils::GetOutput(outs["Ln2Mean"][0], &Ln2Mean);
  paddle::Tensor Ln2Variance;
  egr::EagerUtils::GetOutput(outs["Ln2Variance"][0], &Ln2Variance);
  paddle::Tensor Linear1Out;
  egr::EagerUtils::GetOutput(outs["Linear1Out"][0], &Linear1Out);
  paddle::Tensor Ln1Out;
  egr::EagerUtils::GetOutput(outs["Ln1Out"][0], &Ln1Out);
  paddle::Tensor Dropout1Out;
  egr::EagerUtils::GetOutput(outs["Dropout1Out"][0], &Dropout1Out);
  paddle::Tensor Dropout2Out;
  egr::EagerUtils::GetOutput(outs["Dropout2Out"][0], &Dropout2Out);

  {
    phi::RecordEvent node_creation_record_event(
        "fused_feedforward node_creation", phi::TracerEventType::Operator, 1);
    egr::AutogradMeta* p_autograd_Out = egr::EagerUtils::autograd_meta(&Out);
    egr::AutogradMeta* p_autograd_Dropout1Mask =
        egr::EagerUtils::autograd_meta(&Dropout1Mask);
    egr::AutogradMeta* p_autograd_Dropout2Mask =
        egr::EagerUtils::autograd_meta(&Dropout2Mask);
    egr::AutogradMeta* p_autograd_Ln1Mean =
        egr::EagerUtils::autograd_meta(&Ln1Mean);
    egr::AutogradMeta* p_autograd_Ln1Variance =
        egr::EagerUtils::autograd_meta(&Ln1Variance);
    egr::AutogradMeta* p_autograd_Ln2Mean =
        egr::EagerUtils::autograd_meta(&Ln2Mean);
    egr::AutogradMeta* p_autograd_Ln2Variance =
        egr::EagerUtils::autograd_meta(&Ln2Variance);
    egr::AutogradMeta* p_autograd_Linear1Out =
        egr::EagerUtils::autograd_meta(&Linear1Out);
    egr::AutogradMeta* p_autograd_Ln1Out =
        egr::EagerUtils::autograd_meta(&Ln1Out);
    egr::AutogradMeta* p_autograd_Dropout1Out =
        egr::EagerUtils::autograd_meta(&Dropout1Out);
    egr::AutogradMeta* p_autograd_Dropout2Out =
        egr::EagerUtils::autograd_meta(&Dropout2Out);
    if (require_any_grad) {
      VLOG(6) << " Construct Grad for fused_feedforward ";
      egr::EagerUtils::PassStopGradient(false,
                                        p_autograd_Out,
                                        p_autograd_Dropout1Mask,
                                        p_autograd_Dropout2Mask,
                                        p_autograd_Ln1Mean,
                                        p_autograd_Ln1Variance,
                                        p_autograd_Ln2Mean,
                                        p_autograd_Ln2Variance,
                                        p_autograd_Linear1Out,
                                        p_autograd_Ln1Out,
                                        p_autograd_Dropout1Out,
                                        p_autograd_Dropout2Out);
      // Create GradOpNode
      auto grad_node =
          std::shared_ptr<fused_feedforwardGradNodeCompat>(  // NOLINT
              new fused_feedforwardGradNodeCompat(11, 11));

      bool pre_layer_norm = false;
      if (attrs.count("pre_layer_norm")) {
        pre_layer_norm = PADDLE_GET_CONST(bool, attrs.at("pre_layer_norm"));
      }

      // Set Attributes
      grad_node->SetAttrMap(std::move(attrs));
      grad_node->SetDefaultAttrMap(std::move(default_attrs));

      grad_node->SetTensorWrapper_X(X);
      grad_node->SetTensorWrapper_Linear1Weight(Linear1Weight);
      grad_node->SetTensorWrapper_Linear1Bias(Linear1Bias);
      grad_node->SetTensorWrapper_Linear2Weight(Linear2Weight);
      grad_node->SetTensorWrapper_Dropout1Mask(Dropout1Mask);
      grad_node->SetTensorWrapper_Dropout2Mask(Dropout2Mask);
      grad_node->SetTensorWrapper_Linear1Out(Linear1Out);
      grad_node->SetTensorWrapper_Dropout1Out(Dropout1Out);
      grad_node->SetTensorWrapper_Dropout2Out(Dropout2Out);

      grad_node->SetGradOutMeta(X, 0);
      grad_node->SetGradOutMeta(Linear1Weight, 3);
      grad_node->SetGradOutMeta(Linear1Bias, 4);
      grad_node->SetGradOutMeta(Linear2Weight, 5);

      if (pre_layer_norm) {
        grad_node->SetTensorWrapper_Ln1Scale(Ln1Scale);
        grad_node->SetTensorWrapper_Ln1Bias(Ln1Bias);
        grad_node->SetTensorWrapper_Ln1Out(Ln1Out);
        grad_node->SetTensorWrapper_Ln1Mean(Ln1Mean);
        grad_node->SetTensorWrapper_Ln1Variance(Ln1Variance);
        grad_node->SetGradOutMeta(Ln1Scale, 7);
        grad_node->SetGradOutMeta(Ln1Bias, 8);
      } else {
        grad_node->SetTensorWrapper_Ln2Scale(Ln2Scale);
        grad_node->SetGradOutMeta(Ln2Scale, 9);
        grad_node->SetTensorWrapper_Ln2Bias(Ln2Bias);
        grad_node->SetGradOutMeta(Ln2Bias, 10);
        grad_node->SetTensorWrapper_Ln2Mean(Ln2Mean);
        grad_node->SetTensorWrapper_Ln2Variance(Ln2Variance);
      }

      if (Linear2Bias.initialized()) {
        grad_node->SetTensorWrapper_Linear2Bias(Linear2Bias);
        grad_node->SetGradOutMeta(Linear2Bias, 6);
      }

      egr::EagerUtils::SetOutRankWithSlot(p_autograd_Out, 0);
      egr::EagerUtils::SetHistory(p_autograd_Out, grad_node);
      grad_node->SetGradInMeta(Out, 0);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_Dropout1Mask, 1);
      grad_node->SetGradInMeta(Dropout1Mask, 1);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_Dropout2Mask, 2);
      grad_node->SetGradInMeta(Dropout2Mask, 2);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_Ln1Mean, 3);
      grad_node->SetGradInMeta(Ln1Mean, 3);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_Ln1Variance, 4);
      grad_node->SetGradInMeta(Ln1Variance, 4);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_Ln2Mean, 5);
      grad_node->SetGradInMeta(Ln2Mean, 5);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_Ln2Variance, 6);
      grad_node->SetGradInMeta(Ln2Variance, 6);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_Linear1Out, 7);
      grad_node->SetGradInMeta(Linear1Out, 7);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_Ln1Out, 8);
      grad_node->SetGradInMeta(Ln1Out, 8);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_Dropout1Out, 9);
      grad_node->SetGradInMeta(Dropout1Out, 9);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_Dropout2Out, 10);
      grad_node->SetGradInMeta(Dropout2Out, 10);
    }
  }

  return std::make_tuple(Out,
                         Dropout1Mask,
                         Dropout2Mask,
                         Ln1Mean,
                         Ln1Variance,
                         Ln2Mean,
                         Ln2Variance,
                         Linear1Out,
                         Ln1Out,
                         Dropout1Out,
                         Dropout2Out);
}
