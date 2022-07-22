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
#include "paddle/fluid/eager/amp_utils.h"
#include "paddle/fluid/eager/api/manual/fluid_manual/dygraph_forward_api.h"
#include "paddle/fluid/eager/api/manual/fluid_manual/nodes/nodes.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

#pragma GCC diagnostic ignored "-Wunused-variable"

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
fused_gate_attention_dygraph_function(
    const paddle::experimental::Tensor& Query,
    const paddle::experimental::Tensor& Key,
    const paddle::experimental::Tensor& QueryWeight,
    const paddle::experimental::Tensor& KeyWeight,
    const paddle::experimental::Tensor& ValueWeight,
    const paddle::experimental::Tensor& QKVWeight,
    const paddle::experimental::Tensor& NonbatchedBias,
    const paddle::experimental::Tensor& SrcMask,
    const paddle::experimental::Tensor& GateWeight,
    const paddle::experimental::Tensor& GateBias,
    const paddle::experimental::Tensor& OutLinearWeight,
    const paddle::experimental::Tensor& OutLinearBias,
    const paddle::framework::AttributeMap& attr_map) {
  paddle::platform::RecordEvent dygraph_entrance_record_event(
      "fused_gate_attention dygraph",
      paddle::platform::TracerEventType::Operator,
      1);
  VLOG(3) << "Running Eager Forward Op: fused_gate_attention";
  // Dygraph Forward Pass

  if (egr::Controller::Instance().GetAMPLevel() !=
      paddle::imperative::AmpLevel::O0) {
    VLOG(5) << "Check and Prepare For AMP";

    paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                         egr::kSlotSmallVectorSize>
        amp_tensors_vector = {
            {Query}, {SrcMask}, {OutLinearWeight}, {OutLinearBias}};
    if (Key.initialized()) amp_tensors_vector.push_back({Key});
    if (QueryWeight.initialized()) amp_tensors_vector.push_back({QueryWeight});
    if (KeyWeight.initialized()) amp_tensors_vector.push_back({KeyWeight});
    if (ValueWeight.initialized()) amp_tensors_vector.push_back({ValueWeight});
    if (QKVWeight.initialized()) amp_tensors_vector.push_back({QKVWeight});
    if (NonbatchedBias.initialized())
      amp_tensors_vector.push_back({NonbatchedBias});
    if (GateWeight.initialized()) amp_tensors_vector.push_back({GateWeight});
    if (GateBias.initialized()) amp_tensors_vector.push_back({GateBias});

    auto amp_dst_dtype =
        egr::GetAmpDestDtype("fused_gate_attention", amp_tensors_vector);

    auto NEW_Query =
        egr::AmpAutoCast("Query", Query, amp_dst_dtype, "fused_gate_attention");
    auto NEW_SrcMask = egr::AmpAutoCast(
        "SrcMask", SrcMask, amp_dst_dtype, "fused_gate_attention");
    auto NEW_OutLinearWeight = egr::AmpAutoCast("OutLinearWeight",
                                                OutLinearWeight,
                                                amp_dst_dtype,
                                                "fused_gate_attention");
    auto NEW_OutLinearBias = egr::AmpAutoCast(
        "OutLinearBias", OutLinearBias, amp_dst_dtype, "fused_gate_attention");
    auto NEW_Key = ((Key.initialized())
                        ? egr::AmpAutoCast(
                              "Key", Key, amp_dst_dtype, "fused_gate_attention")
                        : Key);
    auto NEW_QueryWeight =
        ((QueryWeight.initialized()) ? egr::AmpAutoCast("QueryWeight",
                                                        QueryWeight,
                                                        amp_dst_dtype,
                                                        "fused_gate_attention")
                                     : QueryWeight);
    auto NEW_KeyWeight =
        ((KeyWeight.initialized()) ? egr::AmpAutoCast("KeyWeight",
                                                      KeyWeight,
                                                      amp_dst_dtype,
                                                      "fused_gate_attention")
                                   : KeyWeight);
    auto NEW_ValueWeight =
        ((ValueWeight.initialized()) ? egr::AmpAutoCast("ValueWeight",
                                                        ValueWeight,
                                                        amp_dst_dtype,
                                                        "fused_gate_attention")
                                     : ValueWeight);
    auto NEW_QKVWeight =
        ((QKVWeight.initialized()) ? egr::AmpAutoCast("QKVWeight",
                                                      QKVWeight,
                                                      amp_dst_dtype,
                                                      "fused_gate_attention")
                                   : QKVWeight);
    auto NEW_NonbatchedBias = ((NonbatchedBias.initialized())
                                   ? egr::AmpAutoCast("NonbatchedBias",
                                                      NonbatchedBias,
                                                      amp_dst_dtype,
                                                      "fused_gate_attention")
                                   : NonbatchedBias);
    auto NEW_GateWeight =
        ((GateWeight.initialized()) ? egr::AmpAutoCast("GateWeight",
                                                       GateWeight,
                                                       amp_dst_dtype,
                                                       "fused_gate_attention")
                                    : GateWeight);
    auto NEW_GateBias =
        ((GateBias.initialized())
             ? egr::AmpAutoCast(
                   "GateBias", GateBias, amp_dst_dtype, "fused_gate_attention")
             : GateBias);

    {
      paddle::imperative::AutoCastGuard guard(
          egr::Controller::Instance().GetCurrentTracer(),
          paddle::imperative::AmpLevel::O0);
      return fused_gate_attention_dygraph_function(NEW_Query,
                                                   NEW_Key,
                                                   NEW_QueryWeight,
                                                   NEW_KeyWeight,
                                                   NEW_ValueWeight,
                                                   NEW_QKVWeight,
                                                   NEW_NonbatchedBias,
                                                   NEW_SrcMask,
                                                   NEW_GateWeight,
                                                   NEW_GateBias,
                                                   NEW_OutLinearWeight,
                                                   NEW_OutLinearBias,
                                                   attr_map);
    }
  }

  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> ins =
      {{"Query", egr::EagerUtils::TrySyncToVars(Query)},
       {"SrcMask", egr::EagerUtils::TrySyncToVars(SrcMask)},
       {"OutLinearWeight", egr::EagerUtils::TrySyncToVars(OutLinearWeight)},
       {"OutLinearBias", egr::EagerUtils::TrySyncToVars(OutLinearBias)}};
  if (Key.initialized()) ins["Key"] = egr::EagerUtils::TrySyncToVars(Key);
  if (QueryWeight.initialized())
    ins["QueryWeight"] = egr::EagerUtils::TrySyncToVars(QueryWeight);
  if (KeyWeight.initialized())
    ins["KeyWeight"] = egr::EagerUtils::TrySyncToVars(KeyWeight);
  if (ValueWeight.initialized())
    ins["ValueWeight"] = egr::EagerUtils::TrySyncToVars(ValueWeight);
  if (QKVWeight.initialized())
    ins["QKVWeight"] = egr::EagerUtils::TrySyncToVars(QKVWeight);
  if (NonbatchedBias.initialized())
    ins["NonbatchedBias"] = egr::EagerUtils::TrySyncToVars(NonbatchedBias);
  if (GateWeight.initialized())
    ins["GateWeight"] = egr::EagerUtils::TrySyncToVars(GateWeight);
  if (GateBias.initialized())
    ins["GateBias"] = egr::EagerUtils::TrySyncToVars(GateBias);

  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> outs =
      {{"QueryTransposeOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"KeyTransposeOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"ValueTransposeOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"QKVTransposeOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"SoftmaxOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"FMHAOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"GateOut",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}},
       {"Out",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}}};

  // Prepare Autograd Meta
  egr::AutogradMeta* p_autograd_Query =
      egr::EagerUtils::nullable_autograd_meta(Query);
  egr::AutogradMeta* p_autograd_Key =
      egr::EagerUtils::nullable_autograd_meta(Key);
  egr::AutogradMeta* p_autograd_QueryWeight =
      egr::EagerUtils::nullable_autograd_meta(QueryWeight);
  egr::AutogradMeta* p_autograd_KeyWeight =
      egr::EagerUtils::nullable_autograd_meta(KeyWeight);
  egr::AutogradMeta* p_autograd_ValueWeight =
      egr::EagerUtils::nullable_autograd_meta(ValueWeight);
  egr::AutogradMeta* p_autograd_QKVWeight =
      egr::EagerUtils::nullable_autograd_meta(QKVWeight);
  egr::AutogradMeta* p_autograd_NonbatchedBias =
      egr::EagerUtils::nullable_autograd_meta(NonbatchedBias);
  egr::AutogradMeta* p_autograd_SrcMask =
      egr::EagerUtils::nullable_autograd_meta(SrcMask);
  egr::AutogradMeta* p_autograd_GateWeight =
      egr::EagerUtils::nullable_autograd_meta(GateWeight);
  egr::AutogradMeta* p_autograd_GateBias =
      egr::EagerUtils::nullable_autograd_meta(GateBias);
  egr::AutogradMeta* p_autograd_OutLinearWeight =
      egr::EagerUtils::nullable_autograd_meta(OutLinearWeight);
  egr::AutogradMeta* p_autograd_OutLinearBias =
      egr::EagerUtils::nullable_autograd_meta(OutLinearBias);

  bool trace_backward = egr::Controller::Instance().HasGrad();

  bool require_any_grad =
      egr::EagerUtils::ComputeRequireGrad(trace_backward,
                                          p_autograd_Query,
                                          p_autograd_Key,
                                          p_autograd_QueryWeight,
                                          p_autograd_KeyWeight,
                                          p_autograd_ValueWeight,
                                          p_autograd_QKVWeight,
                                          p_autograd_NonbatchedBias,
                                          p_autograd_SrcMask,
                                          p_autograd_GateWeight,
                                          p_autograd_GateBias,
                                          p_autograd_OutLinearWeight,
                                          p_autograd_OutLinearBias);

  paddle::framework::AttributeMap attrs = attr_map;
  paddle::framework::AttributeMap default_attrs;
  egr::Controller::Instance().GetCurrentTracer()->TraceOp(
      "fused_gate_attention",
      ins,
      outs,
      attrs,
      egr::Controller::Instance().GetExpectedPlace(),
      &default_attrs,
      true,
      {});

  paddle::experimental::Tensor QueryTransposeOut;
  egr::EagerUtils::GetOutput(outs["QueryTransposeOut"][0], &QueryTransposeOut);
  paddle::experimental::Tensor KeyTransposeOut;
  egr::EagerUtils::GetOutput(outs["KeyTransposeOut"][0], &KeyTransposeOut);
  paddle::experimental::Tensor ValueTransposeOut;
  egr::EagerUtils::GetOutput(outs["ValueTransposeOut"][0], &ValueTransposeOut);
  paddle::experimental::Tensor QKVTransposeOut;
  egr::EagerUtils::GetOutput(outs["QKVTransposeOut"][0], &QKVTransposeOut);
  paddle::experimental::Tensor SoftmaxOut;
  egr::EagerUtils::GetOutput(outs["SoftmaxOut"][0], &SoftmaxOut);
  paddle::experimental::Tensor FMHAOut;
  egr::EagerUtils::GetOutput(outs["FMHAOut"][0], &FMHAOut);
  paddle::experimental::Tensor GateOut;
  egr::EagerUtils::GetOutput(outs["GateOut"][0], &GateOut);
  paddle::experimental::Tensor Out;
  egr::EagerUtils::GetOutput(outs["Out"][0], &Out);

  {
    paddle::platform::RecordEvent node_creation_record_event(
        "fused_gate_attention node_creation",
        paddle::platform::TracerEventType::Operator,
        1);
    egr::AutogradMeta* p_autograd_QueryTransposeOut =
        egr::EagerUtils::autograd_meta(&QueryTransposeOut);
    egr::AutogradMeta* p_autograd_KeyTransposeOut =
        egr::EagerUtils::autograd_meta(&KeyTransposeOut);
    egr::AutogradMeta* p_autograd_ValueTransposeOut =
        egr::EagerUtils::autograd_meta(&ValueTransposeOut);
    egr::AutogradMeta* p_autograd_QKVTransposeOut =
        egr::EagerUtils::autograd_meta(&QKVTransposeOut);
    egr::AutogradMeta* p_autograd_SoftmaxOut =
        egr::EagerUtils::autograd_meta(&SoftmaxOut);
    egr::AutogradMeta* p_autograd_FMHAOut =
        egr::EagerUtils::autograd_meta(&FMHAOut);
    egr::AutogradMeta* p_autograd_GateOut =
        egr::EagerUtils::autograd_meta(&GateOut);
    egr::AutogradMeta* p_autograd_Out = egr::EagerUtils::autograd_meta(&Out);
    if (require_any_grad) {
      VLOG(6) << " Construct Grad for fused_gate_attention ";
      egr::EagerUtils::PassStopGradient(false,
                                        p_autograd_QueryTransposeOut,
                                        p_autograd_KeyTransposeOut,
                                        p_autograd_ValueTransposeOut,
                                        p_autograd_QKVTransposeOut,
                                        p_autograd_SoftmaxOut,
                                        p_autograd_FMHAOut,
                                        p_autograd_GateOut,
                                        p_autograd_Out);
      // Create GradOpNode
      auto grad_node = std::shared_ptr<fused_gate_attentionGradNodeCompat>(
          new fused_gate_attentionGradNodeCompat(8, 12));

      bool merge_qkv = true;
      if (attrs.count("merge_qkv")) {
        merge_qkv = PADDLE_GET_CONST(bool, attrs.at("merge_qkv"));
      }

      bool has_gating = true;
      if (attrs.count("has_gating")) {
        has_gating = PADDLE_GET_CONST(bool, attrs.at("has_gating"));
      }

      // Set Attributes
      grad_node->SetAttrMap(std::move(attrs));
      grad_node->SetDefaultAttrMap(std::move(default_attrs));

      grad_node->SetTensorWrapperFMHAOut(FMHAOut);
      grad_node->SetTensorWrapperQuery(Query);
      grad_node->SetTensorWrapperSoftmaxOut(SoftmaxOut);
      grad_node->SetTensorWrapperOutLinearBias(OutLinearBias);
      grad_node->SetTensorWrapperOutLinearWeight(OutLinearWeight);

      grad_node->SetGradOutMeta(Query, 0);
      grad_node->SetGradOutMeta(OutLinearWeight, 10);
      grad_node->SetGradOutMeta(OutLinearBias, 11);

      if (merge_qkv) {
        grad_node->SetTensorWrapperQKVTransposeOut(QKVTransposeOut);
        grad_node->SetTensorWrapperQKVWeight(QKVWeight);
        grad_node->SetGradOutMeta(QKVWeight, 5);
      } else {
        grad_node->SetTensorWrapperKey(Key);
        grad_node->SetTensorWrapperQueryWeight(QueryWeight);
        grad_node->SetTensorWrapperKeyWeight(KeyWeight);
        grad_node->SetTensorWrapperValueWeight(ValueWeight);
        grad_node->SetTensorWrapperQueryTransposeOut(QueryTransposeOut);
        grad_node->SetTensorWrapperKeyTransposeOut(KeyTransposeOut);
        grad_node->SetTensorWrapperValueTransposeOut(ValueTransposeOut);

        grad_node->SetGradOutMeta(Key, 1);
        grad_node->SetGradOutMeta(QueryWeight, 2);
        grad_node->SetGradOutMeta(KeyWeight, 3);
        grad_node->SetGradOutMeta(ValueWeight, 4);
      }

      if (has_gating) {
        grad_node->SetTensorWrapperGateWeight(GateWeight);
        grad_node->SetGradOutMeta(GateWeight, 8);
        grad_node->SetTensorWrapperGateBias(GateBias);
        grad_node->SetGradOutMeta(GateBias, 9);
        grad_node->SetTensorWrapperGateOut(GateOut);
      }

      if (NonbatchedBias.initialized()) {
        grad_node->SetTensorWrapperNonbatchedBias(NonbatchedBias);
        grad_node->SetGradOutMeta(NonbatchedBias, 6);
      }

      egr::EagerUtils::SetOutRankWithSlot(p_autograd_QueryTransposeOut, 0);
      grad_node->SetGradInMeta(QueryTransposeOut, 0);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_KeyTransposeOut, 1);
      grad_node->SetGradInMeta(KeyTransposeOut, 1);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_ValueTransposeOut, 2);
      grad_node->SetGradInMeta(ValueTransposeOut, 2);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_QKVTransposeOut, 3);
      grad_node->SetGradInMeta(QKVTransposeOut, 3);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_SoftmaxOut, 4);
      grad_node->SetGradInMeta(SoftmaxOut, 4);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_FMHAOut, 5);
      grad_node->SetGradInMeta(FMHAOut, 5);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_GateOut, 6);
      grad_node->SetGradInMeta(GateOut, 6);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_Out, 7);
      egr::EagerUtils::SetHistory(p_autograd_Out, grad_node);
      grad_node->SetGradInMeta(Out, 7);
      egr::EagerUtils::CheckAndRetainGrad(Out);
    }
  }

  return std::make_tuple(QueryTransposeOut,
                         KeyTransposeOut,
                         ValueTransposeOut,
                         QKVTransposeOut,
                         SoftmaxOut,
                         FMHAOut,
                         GateOut,
                         Out);
}
