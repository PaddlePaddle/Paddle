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
fused_attention_dygraph_function(
    const paddle::Tensor& X,
    const paddle::Tensor& LnScale,
    const paddle::Tensor& LnBias,
    const paddle::Tensor& QKVW,
    const paddle::Tensor& QKVBias,
    const paddle::Tensor& CacheKV,
    const paddle::Tensor& SrcMask,
    const paddle::Tensor& OutLinearW,
    const paddle::Tensor& OutLinearBias,
    const paddle::Tensor& Ln2Scale,
    const paddle::Tensor& Ln2Bias,
    const paddle::Tensor& Seed1,
    const paddle::Tensor& DropoutSeed,
    const paddle::framework::AttributeMap& attr_map) {
  paddle::platform::RecordEvent dygraph_entrance_record_event(
      "fused_attention dygraph",
      paddle::platform::TracerEventType::Operator,
      1);
  VLOG(3) << "Running Eager Forward Op: fused_attention";
  // Dygraph Forward Pass

  if (egr::Controller::Instance().GetAMPLevel() !=
      paddle::imperative::AmpLevel::O0) {
    VLOG(5) << "Check and Prepare For AMP";

    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        amp_tensors_vector = {{X}, {QKVW}, {OutLinearW}};
    if (LnScale.initialized()) amp_tensors_vector.push_back({LnScale});
    if (LnBias.initialized()) amp_tensors_vector.push_back({LnBias});
    if (QKVBias.initialized()) amp_tensors_vector.push_back({QKVBias});
    if (CacheKV.initialized()) amp_tensors_vector.push_back({CacheKV});
    if (SrcMask.initialized()) amp_tensors_vector.push_back({SrcMask});
    if (OutLinearBias.initialized())
      amp_tensors_vector.push_back({OutLinearBias});
    if (Ln2Scale.initialized()) amp_tensors_vector.push_back({Ln2Scale});
    if (Ln2Bias.initialized()) amp_tensors_vector.push_back({Ln2Bias});

    auto amp_dst_dtype =
        egr::GetAmpDestDtype("fused_attention", amp_tensors_vector);

    auto NEW_X = egr::AmpAutoCast("X", X, amp_dst_dtype, "fused_attention");
    auto NEW_QKVW =
        egr::AmpAutoCast("QKVW", QKVW, amp_dst_dtype, "fused_attention");
    auto NEW_OutLinearW = egr::AmpAutoCast(
        "OutLinearW", OutLinearW, amp_dst_dtype, "fused_attention");
    auto NEW_LnScale =
        ((LnScale.initialized())
             ? egr::AmpAutoCast(
                   "LnScale", LnScale, amp_dst_dtype, "fused_attention")
             : LnScale);
    auto NEW_LnBias =
        ((LnBias.initialized())
             ? egr::AmpAutoCast(
                   "LnBias", LnBias, amp_dst_dtype, "fused_attention")
             : LnBias);
    auto NEW_QKVBias =
        ((QKVBias.initialized())
             ? egr::AmpAutoCast(
                   "QKVBias", QKVBias, amp_dst_dtype, "fused_attention")
             : QKVBias);
    auto NEW_CacheKV =
        ((CacheKV.initialized())
             ? egr::AmpAutoCast(
                   "CacheKV", CacheKV, amp_dst_dtype, "fused_attention")
             : CacheKV);
    auto NEW_SrcMask =
        ((SrcMask.initialized())
             ? egr::AmpAutoCast(
                   "SrcMask", SrcMask, amp_dst_dtype, "fused_attention")
             : SrcMask);
    auto NEW_OutLinearBias =
        ((OutLinearBias.initialized()) ? egr::AmpAutoCast("OutLinearBias",
                                                          OutLinearBias,
                                                          amp_dst_dtype,
                                                          "fused_attention")
                                       : OutLinearBias);
    auto NEW_Ln2Scale =
        ((Ln2Scale.initialized())
             ? egr::AmpAutoCast(
                   "Ln2Scale", Ln2Scale, amp_dst_dtype, "fused_attention")
             : Ln2Scale);
    auto NEW_Ln2Bias =
        ((Ln2Bias.initialized())
             ? egr::AmpAutoCast(
                   "Ln2Bias", Ln2Bias, amp_dst_dtype, "fused_attention")
             : Ln2Bias);

    {
      paddle::imperative::AutoCastGuard guard(
          egr::Controller::Instance().GetCurrentTracer(),
          paddle::imperative::AmpLevel::O0);
      return fused_attention_dygraph_function(NEW_X,
                                              NEW_LnScale,
                                              NEW_LnBias,
                                              NEW_QKVW,
                                              NEW_QKVBias,
                                              NEW_CacheKV,
                                              NEW_SrcMask,
                                              NEW_OutLinearW,
                                              NEW_OutLinearBias,
                                              NEW_Ln2Scale,
                                              NEW_Ln2Bias,
                                              Seed1,
                                              DropoutSeed,
                                              attr_map);
    }
  }

  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> ins =
      {{"X", egr::EagerUtils::TrySyncToVars(X)},
       {"QKVW", egr::EagerUtils::TrySyncToVars(QKVW)},
       {"OutLinearW", egr::EagerUtils::TrySyncToVars(OutLinearW)}};
  if (LnScale.initialized())
    ins["LnScale"] = egr::EagerUtils::TrySyncToVars(LnScale);
  if (LnBias.initialized())
    ins["LnBias"] = egr::EagerUtils::TrySyncToVars(LnBias);
  if (QKVBias.initialized())
    ins["QKVBias"] = egr::EagerUtils::TrySyncToVars(QKVBias);
  if (CacheKV.initialized())
    ins["CacheKV"] = egr::EagerUtils::TrySyncToVars(CacheKV);
  if (SrcMask.initialized())
    ins["SrcMask"] = egr::EagerUtils::TrySyncToVars(SrcMask);
  if (OutLinearBias.initialized())
    ins["OutLinearBias"] = egr::EagerUtils::TrySyncToVars(OutLinearBias);
  if (Ln2Scale.initialized())
    ins["Ln2Scale"] = egr::EagerUtils::TrySyncToVars(Ln2Scale);
  if (Ln2Bias.initialized())
    ins["Ln2Bias"] = egr::EagerUtils::TrySyncToVars(Ln2Bias);
  if (Seed1.initialized()) ins["Seed1"] = egr::EagerUtils::TrySyncToVars(Seed1);
  if (DropoutSeed.initialized())
    ins["DropoutSeed"] = egr::EagerUtils::TrySyncToVars(DropoutSeed);

  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> outs =
      {{"LnMean",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"LnVariance",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"LnOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"QKVOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"QKVBiasOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"TransposeOut2",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"QKOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"QKTVOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"SoftmaxOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"AttnDropoutMaskOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"AttnDropoutOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"SrcMaskOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"FMHAOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"OutLinearOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"DropoutMaskOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"Ln2Mean",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"Ln2Variance",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"BiasDropoutResidualOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"CacheKVOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"Y",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}}};

  // Prepare Autograd Meta
  egr::AutogradMeta* p_autograd_X = egr::EagerUtils::nullable_autograd_meta(X);
  egr::AutogradMeta* p_autograd_LnScale =
      egr::EagerUtils::nullable_autograd_meta(LnScale);
  egr::AutogradMeta* p_autograd_LnBias =
      egr::EagerUtils::nullable_autograd_meta(LnBias);
  egr::AutogradMeta* p_autograd_QKVW =
      egr::EagerUtils::nullable_autograd_meta(QKVW);
  egr::AutogradMeta* p_autograd_QKVBias =
      egr::EagerUtils::nullable_autograd_meta(QKVBias);
  egr::AutogradMeta* p_autograd_CacheKV =
      egr::EagerUtils::nullable_autograd_meta(CacheKV);
  egr::AutogradMeta* p_autograd_SrcMask =
      egr::EagerUtils::nullable_autograd_meta(SrcMask);
  egr::AutogradMeta* p_autograd_OutLinearW =
      egr::EagerUtils::nullable_autograd_meta(OutLinearW);
  egr::AutogradMeta* p_autograd_OutLinearBias =
      egr::EagerUtils::nullable_autograd_meta(OutLinearBias);
  egr::AutogradMeta* p_autograd_Ln2Scale =
      egr::EagerUtils::nullable_autograd_meta(Ln2Scale);
  egr::AutogradMeta* p_autograd_Ln2Bias =
      egr::EagerUtils::nullable_autograd_meta(Ln2Bias);
  egr::AutogradMeta* p_autograd_Seed1 =
      egr::EagerUtils::nullable_autograd_meta(Seed1);
  egr::AutogradMeta* p_autograd_DropoutSeed =
      egr::EagerUtils::nullable_autograd_meta(DropoutSeed);

  bool trace_backward = egr::Controller::Instance().HasGrad();

  bool require_any_grad =
      egr::EagerUtils::ComputeRequireGrad(trace_backward,
                                          p_autograd_X,
                                          p_autograd_LnScale,
                                          p_autograd_LnBias,
                                          p_autograd_QKVW,
                                          p_autograd_QKVBias,
                                          p_autograd_CacheKV,
                                          p_autograd_SrcMask,
                                          p_autograd_OutLinearW,
                                          p_autograd_OutLinearBias,
                                          p_autograd_Ln2Scale,
                                          p_autograd_Ln2Bias,
                                          p_autograd_Seed1,
                                          p_autograd_DropoutSeed);

  paddle::framework::AttributeMap attrs = attr_map;
  paddle::framework::AttributeMap default_attrs;
  egr::Controller::Instance().GetCurrentTracer()->TraceOp(
      "fused_attention",
      ins,
      outs,
      attrs,
      egr::Controller::Instance().GetExpectedPlace(),
      &default_attrs,
      true,
      {});

  paddle::Tensor LnMean;
  egr::EagerUtils::GetOutput(outs["LnMean"][0], &LnMean);
  paddle::Tensor LnVariance;
  egr::EagerUtils::GetOutput(outs["LnVariance"][0], &LnVariance);
  paddle::Tensor LnOut;
  egr::EagerUtils::GetOutput(outs["LnOut"][0], &LnOut);
  paddle::Tensor QKVOut;
  egr::EagerUtils::GetOutput(outs["QKVOut"][0], &QKVOut);
  paddle::Tensor QKVBiasOut;
  egr::EagerUtils::GetOutput(outs["QKVBiasOut"][0], &QKVBiasOut);
  paddle::Tensor TransposeOut2;
  egr::EagerUtils::GetOutput(outs["TransposeOut2"][0], &TransposeOut2);
  paddle::Tensor QKOut;
  egr::EagerUtils::GetOutput(outs["QKOut"][0], &QKOut);
  paddle::Tensor QKTVOut;
  egr::EagerUtils::GetOutput(outs["QKTVOut"][0], &QKTVOut);
  paddle::Tensor SoftmaxOut;
  egr::EagerUtils::GetOutput(outs["SoftmaxOut"][0], &SoftmaxOut);
  paddle::Tensor AttnDropoutMaskOut;
  egr::EagerUtils::GetOutput(outs["AttnDropoutMaskOut"][0],
                             &AttnDropoutMaskOut);
  paddle::Tensor AttnDropoutOut;
  egr::EagerUtils::GetOutput(outs["AttnDropoutOut"][0], &AttnDropoutOut);
  paddle::Tensor SrcMaskOut;
  egr::EagerUtils::GetOutput(outs["SrcMaskOut"][0], &SrcMaskOut);
  paddle::Tensor FMHAOut;
  egr::EagerUtils::GetOutput(outs["FMHAOut"][0], &FMHAOut);
  paddle::Tensor OutLinearOut;
  egr::EagerUtils::GetOutput(outs["OutLinearOut"][0], &OutLinearOut);
  paddle::Tensor DropoutMaskOut;
  egr::EagerUtils::GetOutput(outs["DropoutMaskOut"][0], &DropoutMaskOut);
  paddle::Tensor Ln2Mean;
  egr::EagerUtils::GetOutput(outs["Ln2Mean"][0], &Ln2Mean);
  paddle::Tensor Ln2Variance;
  egr::EagerUtils::GetOutput(outs["Ln2Variance"][0], &Ln2Variance);
  paddle::Tensor BiasDropoutResidualOut;
  egr::EagerUtils::GetOutput(outs["BiasDropoutResidualOut"][0],
                             &BiasDropoutResidualOut);
  paddle::Tensor CacheKVOut;
  egr::EagerUtils::GetOutput(outs["CacheKVOut"][0], &CacheKVOut);
  paddle::Tensor Y;
  egr::EagerUtils::GetOutput(outs["Y"][0], &Y);

  {
    paddle::platform::RecordEvent node_creation_record_event(
        "fused_attention node_creation",
        paddle::platform::TracerEventType::Operator,
        1);
    egr::AutogradMeta* p_autograd_LnMean =
        egr::EagerUtils::autograd_meta(&LnMean);
    egr::AutogradMeta* p_autograd_LnVariance =
        egr::EagerUtils::autograd_meta(&LnVariance);
    egr::AutogradMeta* p_autograd_LnOut =
        egr::EagerUtils::autograd_meta(&LnOut);
    egr::AutogradMeta* p_autograd_QKVOut =
        egr::EagerUtils::autograd_meta(&QKVOut);
    egr::AutogradMeta* p_autograd_QKVBiasOut =
        egr::EagerUtils::autograd_meta(&QKVBiasOut);
    egr::AutogradMeta* p_autograd_TransposeOut2 =
        egr::EagerUtils::autograd_meta(&TransposeOut2);
    egr::AutogradMeta* p_autograd_QKOut =
        egr::EagerUtils::autograd_meta(&QKOut);
    egr::AutogradMeta* p_autograd_QKTVOut =
        egr::EagerUtils::autograd_meta(&QKTVOut);
    egr::AutogradMeta* p_autograd_SoftmaxOut =
        egr::EagerUtils::autograd_meta(&SoftmaxOut);
    egr::AutogradMeta* p_autograd_AttnDropoutMaskOut =
        egr::EagerUtils::autograd_meta(&AttnDropoutMaskOut);
    egr::AutogradMeta* p_autograd_AttnDropoutOut =
        egr::EagerUtils::autograd_meta(&AttnDropoutOut);
    egr::AutogradMeta* p_autograd_SrcMaskOut =
        egr::EagerUtils::autograd_meta(&SrcMaskOut);
    egr::AutogradMeta* p_autograd_FMHAOut =
        egr::EagerUtils::autograd_meta(&FMHAOut);
    egr::AutogradMeta* p_autograd_OutLinearOut =
        egr::EagerUtils::autograd_meta(&OutLinearOut);
    egr::AutogradMeta* p_autograd_DropoutMaskOut =
        egr::EagerUtils::autograd_meta(&DropoutMaskOut);
    egr::AutogradMeta* p_autograd_Ln2Mean =
        egr::EagerUtils::autograd_meta(&Ln2Mean);
    egr::AutogradMeta* p_autograd_Ln2Variance =
        egr::EagerUtils::autograd_meta(&Ln2Variance);
    egr::AutogradMeta* p_autograd_BiasDropoutResidualOut =
        egr::EagerUtils::autograd_meta(&BiasDropoutResidualOut);
    egr::AutogradMeta* p_autograd_CacheKVOut =
        egr::EagerUtils::autograd_meta(&CacheKVOut);
    egr::AutogradMeta* p_autograd_Y = egr::EagerUtils::autograd_meta(&Y);
    if (require_any_grad) {
      VLOG(6) << " Construct Grad for fused_attention ";
      egr::EagerUtils::PassStopGradient(false, p_autograd_Y);
      egr::EagerUtils::PassStopGradient(true,
                                        p_autograd_LnOut,
                                        p_autograd_QKVOut,
                                        p_autograd_QKVBiasOut,
                                        p_autograd_TransposeOut2,
                                        p_autograd_QKOut,
                                        p_autograd_QKTVOut,
                                        p_autograd_SoftmaxOut,
                                        p_autograd_AttnDropoutMaskOut,
                                        p_autograd_AttnDropoutOut,
                                        p_autograd_SrcMaskOut,
                                        p_autograd_FMHAOut,
                                        p_autograd_OutLinearOut,
                                        p_autograd_DropoutMaskOut,
                                        p_autograd_Ln2Mean,
                                        p_autograd_Ln2Variance,
                                        p_autograd_BiasDropoutResidualOut,
                                        p_autograd_CacheKVOut);
      // Create GradOpNode
      auto grad_node = std::shared_ptr<fused_attentionGradNodeCompat>(
          new fused_attentionGradNodeCompat(20, 13));

      bool pre_layer_norm = false;
      if (attrs.count("pre_layer_norm")) {
        pre_layer_norm = PADDLE_GET_CONST(bool, attrs.at("pre_layer_norm"));
      }

      // Set Attributes
      grad_node->SetAttrMap(std::move(attrs));
      grad_node->SetDefaultAttrMap(std::move(default_attrs));

      grad_node->SetTensorWrapperX(X);
      grad_node->SetTensorWrapperQKVW(QKVW);
      grad_node->SetTensorWrapperOutLinearW(OutLinearW);
      grad_node->SetTensorWrapperQKVOut(QKVOut);
      grad_node->SetTensorWrapperTransposeOut2(TransposeOut2);
      grad_node->SetTensorWrapperQKOut(QKOut);
      grad_node->SetTensorWrapperQKTVOut(QKTVOut);
      grad_node->SetTensorWrapperSoftmaxOut(SoftmaxOut);
      grad_node->SetTensorWrapperAttnDropoutMaskOut(AttnDropoutMaskOut);
      grad_node->SetTensorWrapperAttnDropoutOut(AttnDropoutOut);
      grad_node->SetTensorWrapperFMHAOut(FMHAOut);
      grad_node->SetTensorWrapperOutLinearOut(OutLinearOut);
      grad_node->SetTensorWrapperDropoutMaskOut(DropoutMaskOut);

      grad_node->SetGradOutMeta(X, 0);
      grad_node->SetGradOutMeta(QKVW, 3);
      grad_node->SetGradOutMeta(OutLinearW, 7);

      if (Seed1.initialized()) {
        grad_node->SetTensorWrapperSeed1(Seed1);
        grad_node->SetGradOutMeta(Seed1, 11);
      }
      if (DropoutSeed.initialized()) {
        grad_node->SetTensorWrapperDropoutSeed(DropoutSeed);
        grad_node->SetGradOutMeta(DropoutSeed, 12);
      }
      if (QKVBias.initialized()) {
        grad_node->SetTensorWrapperQKVBias(QKVBias);
        grad_node->SetTensorWrapperQKVBiasOut(QKVBiasOut);
        grad_node->SetGradOutMeta(QKVBias, 4);
      }

      if (SrcMask.initialized()) {
        grad_node->SetTensorWrapperSrcMask(SrcMask);
        grad_node->SetTensorWrapperSrcMaskOut(SrcMaskOut);
      }

      if (OutLinearBias.initialized()) {
        grad_node->SetTensorWrapperOutLinearBias(OutLinearBias);
        grad_node->SetGradOutMeta(OutLinearBias, 8);
      }

      if (pre_layer_norm) {
        if (LnScale.initialized()) {
          grad_node->SetTensorWrapperLnScale(LnScale);
          grad_node->SetGradOutMeta(LnScale, 1);
        }
        if (LnBias.initialized()) {
          grad_node->SetTensorWrapperLnBias(LnBias);
          grad_node->SetGradOutMeta(LnBias, 2);
        }
        if (LnOut.initialized()) {
          grad_node->SetTensorWrapperLnOut(LnOut);
        }
        if (LnMean.initialized()) {
          grad_node->SetTensorWrapperLnMean(LnMean);
        }
        if (LnVariance.initialized()) {
          grad_node->SetTensorWrapperLnVariance(LnVariance);
        }
      } else {
        if (Ln2Scale.initialized()) {
          grad_node->SetTensorWrapperLn2Scale(Ln2Scale);
          grad_node->SetGradOutMeta(Ln2Scale, 9);
        }
        if (Ln2Bias.initialized()) {
          grad_node->SetTensorWrapperLn2Bias(Ln2Bias);
          grad_node->SetGradOutMeta(Ln2Bias, 10);
        }
        grad_node->SetTensorWrapperBiasDropoutResidualOut(
            BiasDropoutResidualOut);
        grad_node->SetTensorWrapperLn2Mean(Ln2Mean);
        grad_node->SetTensorWrapperLn2Variance(Ln2Variance);
      }

      egr::EagerUtils::SetOutRankWithSlot(p_autograd_LnMean, 0);
      grad_node->SetGradInMeta(LnMean, 0);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_LnVariance, 1);
      grad_node->SetGradInMeta(LnVariance, 1);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_Ln2Mean, 15);
      grad_node->SetGradInMeta(Ln2Mean, 15);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_Ln2Variance, 16);
      grad_node->SetGradInMeta(Ln2Variance, 16);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_CacheKVOut, 18);
      egr::EagerUtils::SetHistory(p_autograd_CacheKVOut, grad_node);
      grad_node->SetGradInMeta(CacheKVOut, 18);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_Y, 19);
      egr::EagerUtils::SetHistory(p_autograd_Y, grad_node);
      grad_node->SetGradInMeta(Y, 19);
    }
  }

  return std::make_tuple(LnMean,
                         LnVariance,
                         LnOut,
                         QKVOut,
                         QKVBiasOut,
                         TransposeOut2,
                         QKOut,
                         QKTVOut,
                         SoftmaxOut,
                         AttnDropoutMaskOut,
                         AttnDropoutOut,
                         SrcMaskOut,
                         FMHAOut,
                         OutLinearOut,
                         DropoutMaskOut,
                         Ln2Mean,
                         Ln2Variance,
                         BiasDropoutResidualOut,
                         CacheKVOut,
                         Y);
}
