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
    const paddle::framework::AttributeMap& attr_map) {
  phi::RecordEvent dygraph_entrance_record_event(
      "fused_attention dygraph", phi::TracerEventType::Operator, 1);
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

    auto amp_dst_dtype = paddle::imperative::GetAmpDestDtype(
        "fused_attention", amp_tensors_vector);

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
          egr::Controller::Instance().GetCurrentAmpAttrs(),
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
                                          p_autograd_Ln2Bias);

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
    phi::RecordEvent node_creation_record_event(
        "fused_attention node_creation", phi::TracerEventType::Operator, 1);
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
      egr::EagerUtils::PassStopGradient(false,
                                        p_autograd_LnMean,
                                        p_autograd_LnVariance,
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
                                        p_autograd_CacheKVOut,
                                        p_autograd_Y);
      // Create GradOpNode
      auto grad_node =
          std::shared_ptr<fused_attentionGradNodeCompat>(  // NOLINT
              new fused_attentionGradNodeCompat(20, 23));

      bool pre_layer_norm = false;
      if (attrs.count("pre_layer_norm")) {
        pre_layer_norm = PADDLE_GET_CONST(bool, attrs.at("pre_layer_norm"));
      }

      // Set Attributes
      grad_node->SetAttrMap(std::move(attrs));
      grad_node->SetDefaultAttrMap(std::move(default_attrs));

      grad_node->SetTensorWrapper_X(X);
      grad_node->SetTensorWrapper_QKVW(QKVW);
      grad_node->SetTensorWrapper_OutLinearW(OutLinearW);
      grad_node->SetTensorWrapper_QKVOut(QKVOut);
      grad_node->SetTensorWrapper_TransposeOut2(TransposeOut2);
      grad_node->SetTensorWrapper_QKOut(QKOut);
      grad_node->SetTensorWrapper_QKTVOut(QKTVOut);
      grad_node->SetTensorWrapper_SoftmaxOut(SoftmaxOut);
      grad_node->SetTensorWrapper_AttnDropoutMaskOut(AttnDropoutMaskOut);
      grad_node->SetTensorWrapper_AttnDropoutOut(AttnDropoutOut);
      grad_node->SetTensorWrapper_FMHAOut(FMHAOut);
      grad_node->SetTensorWrapper_OutLinearOut(OutLinearOut);
      grad_node->SetTensorWrapper_DropoutMaskOut(DropoutMaskOut);

      grad_node->SetGradOutMeta(X, 0);
      grad_node->SetGradOutMeta(QKVW, 3);
      grad_node->SetGradOutMeta(OutLinearW, 7);

      if (QKVBias.initialized()) {
        grad_node->SetTensorWrapper_QKVBias(QKVBias);
        grad_node->SetTensorWrapper_QKVBiasOut(QKVBiasOut);
        grad_node->SetGradOutMeta(QKVBias, 4);

        auto QKVBiasOut_accumulation_node =
            std::make_shared<egr::GradNodeAccumulation>(p_autograd_QKVBiasOut);
        egr::EagerUtils::SetOutRankWithSlot(p_autograd_QKVBiasOut, 0);
        egr::EagerUtils::SetHistory(p_autograd_QKVBiasOut,
                                    QKVBiasOut_accumulation_node);
        QKVBiasOut_accumulation_node->SetGradInMeta(QKVBiasOut, 0);
        grad_node->SetGradOutMeta(QKVBiasOut, 11);
      }

      if (SrcMask.initialized()) {
        grad_node->SetTensorWrapper_SrcMask(SrcMask);
        grad_node->SetTensorWrapper_SrcMaskOut(SrcMaskOut);

        auto SrcMaskOut_accumulation_node =
            std::make_shared<egr::GradNodeAccumulation>(p_autograd_SrcMaskOut);
        egr::EagerUtils::SetOutRankWithSlot(p_autograd_SrcMaskOut, 0);
        egr::EagerUtils::SetHistory(p_autograd_SrcMaskOut,
                                    SrcMaskOut_accumulation_node);
        SrcMaskOut_accumulation_node->SetGradInMeta(SrcMaskOut, 0);
        grad_node->SetGradOutMeta(SrcMaskOut, 12);
      }

      if (OutLinearBias.initialized()) {
        grad_node->SetTensorWrapper_OutLinearBias(OutLinearBias);
        grad_node->SetGradOutMeta(OutLinearBias, 8);
      }

      if (pre_layer_norm) {
        if (LnScale.initialized()) {
          grad_node->SetTensorWrapper_LnScale(LnScale);
          grad_node->SetGradOutMeta(LnScale, 1);
        }
        if (LnBias.initialized()) {
          grad_node->SetTensorWrapper_LnBias(LnBias);
          grad_node->SetGradOutMeta(LnBias, 2);
        }
        if (LnOut.initialized()) {
          grad_node->SetTensorWrapper_LnOut(LnOut);

          auto LnOut_accumulation_node =
              std::make_shared<egr::GradNodeAccumulation>(p_autograd_LnOut);
          egr::EagerUtils::SetOutRankWithSlot(p_autograd_LnOut, 0);
          egr::EagerUtils::SetHistory(p_autograd_LnOut,
                                      LnOut_accumulation_node);
          LnOut_accumulation_node->SetGradInMeta(LnOut, 0);
          grad_node->SetGradOutMeta(LnOut, 13);
        }
        if (LnMean.initialized()) {
          grad_node->SetTensorWrapper_LnMean(LnMean);
        }
        if (LnVariance.initialized()) {
          grad_node->SetTensorWrapper_LnVariance(LnVariance);
        }
      } else {
        if (Ln2Scale.initialized()) {
          grad_node->SetTensorWrapper_Ln2Scale(Ln2Scale);
          grad_node->SetGradOutMeta(Ln2Scale, 9);
        }
        if (Ln2Bias.initialized()) {
          grad_node->SetTensorWrapper_Ln2Bias(Ln2Bias);
          grad_node->SetGradOutMeta(Ln2Bias, 10);
        }
        grad_node->SetTensorWrapper_BiasDropoutResidualOut(
            BiasDropoutResidualOut);
        grad_node->SetTensorWrapper_Ln2Mean(Ln2Mean);
        grad_node->SetTensorWrapper_Ln2Variance(Ln2Variance);

        auto BiasDropoutResidualOut_accumulation_node =
            std::make_shared<egr::GradNodeAccumulation>(
                p_autograd_BiasDropoutResidualOut);
        egr::EagerUtils::SetOutRankWithSlot(p_autograd_BiasDropoutResidualOut,
                                            0);
        egr::EagerUtils::SetHistory(p_autograd_BiasDropoutResidualOut,
                                    BiasDropoutResidualOut_accumulation_node);
        BiasDropoutResidualOut_accumulation_node->SetGradInMeta(
            BiasDropoutResidualOut, 0);
        grad_node->SetGradOutMeta(BiasDropoutResidualOut, 14);
      }

      egr::EagerUtils::SetOutRankWithSlot(p_autograd_LnMean, 0);
      grad_node->SetGradInMeta(LnMean, 0);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_LnVariance, 1);
      grad_node->SetGradInMeta(LnVariance, 1);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_AttnDropoutMaskOut, 9);
      grad_node->SetGradInMeta(AttnDropoutMaskOut, 9);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_DropoutMaskOut, 14);
      grad_node->SetGradInMeta(DropoutMaskOut, 14);
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
      auto QKVOut_accumulation_node =
          std::make_shared<egr::GradNodeAccumulation>(p_autograd_QKVOut);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_QKVOut, 0);
      egr::EagerUtils::SetHistory(p_autograd_QKVOut, QKVOut_accumulation_node);
      QKVOut_accumulation_node->SetGradInMeta(QKVOut, 0);
      grad_node->SetGradOutMeta(QKVOut, 15);

      auto QKTVOut_accumulation_node =
          std::make_shared<egr::GradNodeAccumulation>(p_autograd_QKTVOut);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_QKTVOut, 0);
      egr::EagerUtils::SetHistory(p_autograd_QKTVOut,
                                  QKTVOut_accumulation_node);
      QKTVOut_accumulation_node->SetGradInMeta(QKTVOut, 0);
      grad_node->SetGradOutMeta(QKTVOut, 16);

      auto TransposeOut2_accumulation_node =
          std::make_shared<egr::GradNodeAccumulation>(p_autograd_TransposeOut2);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_TransposeOut2, 0);
      egr::EagerUtils::SetHistory(p_autograd_TransposeOut2,
                                  TransposeOut2_accumulation_node);
      TransposeOut2_accumulation_node->SetGradInMeta(TransposeOut2, 0);
      grad_node->SetGradOutMeta(TransposeOut2, 17);

      auto QKOut_accumulation_node =
          std::make_shared<egr::GradNodeAccumulation>(p_autograd_QKOut);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_QKOut, 0);
      egr::EagerUtils::SetHistory(p_autograd_QKOut, QKOut_accumulation_node);
      QKOut_accumulation_node->SetGradInMeta(QKOut, 0);
      grad_node->SetGradOutMeta(QKOut, 18);

      auto SoftmaxOut_accumulation_node =
          std::make_shared<egr::GradNodeAccumulation>(p_autograd_SoftmaxOut);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_SoftmaxOut, 0);
      egr::EagerUtils::SetHistory(p_autograd_SoftmaxOut,
                                  SoftmaxOut_accumulation_node);
      SoftmaxOut_accumulation_node->SetGradInMeta(SoftmaxOut, 0);
      grad_node->SetGradOutMeta(SoftmaxOut, 19);

      if (AttnDropoutOut.initialized()) {
        auto AttnDropoutOut_accumulation_node =
            std::make_shared<egr::GradNodeAccumulation>(
                p_autograd_AttnDropoutOut);
        egr::EagerUtils::SetOutRankWithSlot(p_autograd_AttnDropoutOut, 0);
        egr::EagerUtils::SetHistory(p_autograd_AttnDropoutOut,
                                    AttnDropoutOut_accumulation_node);
        AttnDropoutOut_accumulation_node->SetGradInMeta(AttnDropoutOut, 0);
        grad_node->SetGradOutMeta(AttnDropoutOut, 20);
      }

      auto FMHAOut_accumulation_node =
          std::make_shared<egr::GradNodeAccumulation>(p_autograd_FMHAOut);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_FMHAOut, 0);
      egr::EagerUtils::SetHistory(p_autograd_FMHAOut,
                                  FMHAOut_accumulation_node);
      FMHAOut_accumulation_node->SetGradInMeta(FMHAOut, 0);
      grad_node->SetGradOutMeta(FMHAOut, 21);

      auto OutLinearOut_accumulation_node =
          std::make_shared<egr::GradNodeAccumulation>(p_autograd_OutLinearOut);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_OutLinearOut, 0);
      egr::EagerUtils::SetHistory(p_autograd_OutLinearOut,
                                  OutLinearOut_accumulation_node);
      OutLinearOut_accumulation_node->SetGradInMeta(OutLinearOut, 0);
      grad_node->SetGradOutMeta(OutLinearOut, 22);
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
