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

#include "glog/logging.h"
#include "paddle/fluid/eager/api/manual/fluid_manual/nodes/nodes.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/phi/api/all.h"

paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                     egr::kSlotSmallVectorSize>
fused_gate_attentionGradNodeCompat::operator()(
    paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                         egr::kSlotSmallVectorSize>& grads,
    bool create_graph,
    bool is_new_grad) {
  VLOG(3) << "Running Eager Backward Node: fused_gate_attentionGradNodeCompat";

  const auto& out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       egr::kSlotSmallVectorSize>
      outputs(12);
  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       egr::kSlotSmallVectorSize>
      hooked_grads0 =
          fused_gate_attentionGradNodeCompat::ApplyGradientHooks(grads);

  bool merge_qkv = true;
  if (attr_map_.count("merge_qkv")) {
    merge_qkv = PADDLE_GET_CONST(bool, attr_map_.at("merge_qkv"));
  }

  bool has_gating = true;
  if (attr_map_.count("has_gating")) {
    has_gating = PADDLE_GET_CONST(bool, attr_map_.at("has_gating"));
  }

  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> ins0 =
      {{"FMHAOut",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->FMHAOut_))},
       {"Out@GRAD", egr::EagerUtils::TrySyncToVars(hooked_grads0[7])},
       {"OutLinearBias",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->OutLinearBias_))},
       {"OutLinearWeight",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->OutLinearWeight_))},
       {"Query",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->Query_))},
       {"SoftmaxOut",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->SoftmaxOut_))}};
  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> outs0;

  if ((!out_metas[11].empty()) && (!(out_metas[11][0].IsStopGradient()))) {
    outs0.insert({"OutLinearBias@GRAD",
                  {std::make_shared<egr::EagerVariable>(
                      egr::Controller::Instance().GenerateUniqueName())}});
  }
  if ((!out_metas[10].empty()) && (!(out_metas[10][0].IsStopGradient()))) {
    outs0.insert({"OutLinearWeight@GRAD",
                  {std::make_shared<egr::EagerVariable>(
                      egr::Controller::Instance().GenerateUniqueName())}});
  }
  if ((!out_metas[0].empty()) && (!(out_metas[0][0].IsStopGradient()))) {
    outs0.insert({"Query@GRAD",
                  {std::make_shared<egr::EagerVariable>(
                      egr::Controller::Instance().GenerateUniqueName())}});
  }

  if (merge_qkv) {
    auto QKVTransposeOut =
        egr::EagerUtils::RecoverTensorWrapper(&this->QKVTransposeOut_);
    if (QKVTransposeOut.defined())
      ins0["QKVTransposeOut"] = egr::EagerUtils::TrySyncToVars(QKVTransposeOut);
    auto QKVWeight = egr::EagerUtils::RecoverTensorWrapper(&this->QKVWeight_);
    if (QKVWeight.defined())
      ins0["QKVWeight"] = egr::EagerUtils::TrySyncToVars(QKVWeight);
    if (QKVWeight.defined() && (!out_metas[5].empty()) &&
        (!out_metas[5][0].IsStopGradient()))
      outs0["QKVWeight@GRAD"] = {std::make_shared<egr::EagerVariable>(
          egr::Controller::Instance().GenerateUniqueName())};
  } else {
    auto Key = egr::EagerUtils::RecoverTensorWrapper(&this->Key_);
    if (Key.defined()) ins0["Key"] = egr::EagerUtils::TrySyncToVars(Key);
    auto QueryWeight =
        egr::EagerUtils::RecoverTensorWrapper(&this->QueryWeight_);
    if (QueryWeight.defined())
      ins0["QueryWeight"] = egr::EagerUtils::TrySyncToVars(QueryWeight);
    auto KeyWeight = egr::EagerUtils::RecoverTensorWrapper(&this->KeyWeight_);
    if (KeyWeight.defined())
      ins0["KeyWeight"] = egr::EagerUtils::TrySyncToVars(KeyWeight);
    auto ValueWeight =
        egr::EagerUtils::RecoverTensorWrapper(&this->ValueWeight_);
    if (ValueWeight.defined())
      ins0["ValueWeight"] = egr::EagerUtils::TrySyncToVars(ValueWeight);
    auto QueryTransposeOut =
        egr::EagerUtils::RecoverTensorWrapper(&this->QueryTransposeOut_);
    if (QueryTransposeOut.defined())
      ins0["QueryTransposeOut"] =
          egr::EagerUtils::TrySyncToVars(QueryTransposeOut);
    auto KeyTransposeOut =
        egr::EagerUtils::RecoverTensorWrapper(&this->KeyTransposeOut_);
    if (KeyTransposeOut.defined())
      ins0["KeyTransposeOut"] = egr::EagerUtils::TrySyncToVars(KeyTransposeOut);
    auto ValueTransposeOut =
        egr::EagerUtils::RecoverTensorWrapper(&this->ValueTransposeOut_);
    if (ValueTransposeOut.defined())
      ins0["ValueTransposeOut"] =
          egr::EagerUtils::TrySyncToVars(ValueTransposeOut);

    if (Key.defined() && (!out_metas[1].empty()) &&
        (!out_metas[1][0].IsStopGradient()))
      outs0["Key@GRAD"] = {std::make_shared<egr::EagerVariable>(
          egr::Controller::Instance().GenerateUniqueName())};
    if (QueryWeight.defined() && (!out_metas[2].empty()) &&
        (!out_metas[2][0].IsStopGradient()))
      outs0["QueryWeight@GRAD"] = {std::make_shared<egr::EagerVariable>(
          egr::Controller::Instance().GenerateUniqueName())};
    if (KeyWeight.defined() && (!out_metas[3].empty()) &&
        (!out_metas[3][0].IsStopGradient()))
      outs0["KeyWeight@GRAD"] = {std::make_shared<egr::EagerVariable>(
          egr::Controller::Instance().GenerateUniqueName())};
    if (ValueWeight.defined() && (!out_metas[4].empty()) &&
        (!out_metas[4][0].IsStopGradient()))
      outs0["ValueWeight@GRAD"] = {std::make_shared<egr::EagerVariable>(
          egr::Controller::Instance().GenerateUniqueName())};
  }

  if (has_gating) {
    auto GateBias = egr::EagerUtils::RecoverTensorWrapper(&this->GateBias_);
    if (GateBias.defined())
      ins0["GateBias"] = egr::EagerUtils::TrySyncToVars(GateBias);
    auto GateWeight = egr::EagerUtils::RecoverTensorWrapper(&this->GateWeight_);
    if (GateWeight.defined())
      ins0["GateWeight"] = egr::EagerUtils::TrySyncToVars(GateWeight);
    auto GateOut = egr::EagerUtils::RecoverTensorWrapper(&this->GateOut_);
    if (GateOut.defined())
      ins0["GateOut"] = egr::EagerUtils::TrySyncToVars(GateOut);
    if (GateBias.defined() && (!out_metas[9].empty()) &&
        (!out_metas[9][0].IsStopGradient()))
      outs0["GateBias@GRAD"] = {std::make_shared<egr::EagerVariable>(
          egr::Controller::Instance().GenerateUniqueName())};
    if (GateWeight.defined() && (!out_metas[8].empty()) &&
        (!out_metas[8][0].IsStopGradient()))
      outs0["GateWeight@GRAD"] = {std::make_shared<egr::EagerVariable>(
          egr::Controller::Instance().GenerateUniqueName())};
  }

  auto NonbatchedBias =
      egr::EagerUtils::RecoverTensorWrapper(&this->NonbatchedBias_);
  if (NonbatchedBias.defined()) {
    ins0["NonbatchedBias"] = egr::EagerUtils::TrySyncToVars(NonbatchedBias);
    if ((!out_metas[6].empty()) && (!out_metas[6][0].IsStopGradient()))
      outs0["NonbatchedBias@GRAD"] = {std::make_shared<egr::EagerVariable>(
          egr::Controller::Instance().GenerateUniqueName())};
  }

  auto& attrs_map0 = this->attr_map_;
  // Pass the entire attribute map to TraceOp
  // The underlying kernel will pickup whatever attribute they need at runtime
  egr::Controller::Instance().GetCurrentTracer()->TraceOp(
      "fused_gate_attention_grad",
      ins0,
      outs0,
      attrs_map0,
      egr::Controller::Instance().GetExpectedPlace(),
      &this->default_attr_map_,
      false,
      {});

  if (outs0.find("Query@GRAD") != outs0.end()) {
    outputs[0] = egr::EagerUtils::GetOutputs(outs0["Query@GRAD"]);
  }
  if (outs0.find("OutLinearBias@GRAD") != outs0.end()) {
    outputs[11] = egr::EagerUtils::GetOutputs(outs0["OutLinearBias@GRAD"]);
  }
  if (outs0.find("OutLinearWeight@GRAD") != outs0.end()) {
    outputs[10] = egr::EagerUtils::GetOutputs(outs0["OutLinearWeight@GRAD"]);
  }

  if (merge_qkv) {
    if (outs0.find("QKVWeight@GRAD") != outs0.end()) {
      outputs[5] = egr::EagerUtils::GetOutputs(outs0["QKVWeight@GRAD"]);
    }
  } else {
    if (outs0.find("Key@GRAD") != outs0.end()) {
      outputs[1] = egr::EagerUtils::GetOutputs(outs0["Key@GRAD"]);
    }
    if (outs0.find("QueryWeight@GRAD") != outs0.end()) {
      outputs[2] = egr::EagerUtils::GetOutputs(outs0["QueryWeight@GRAD"]);
    }
    if (outs0.find("KeyWeight@GRAD") != outs0.end()) {
      outputs[3] = egr::EagerUtils::GetOutputs(outs0["KeyWeight@GRAD"]);
    }
    if (outs0.find("ValueWeight@GRAD") != outs0.end()) {
      outputs[4] = egr::EagerUtils::GetOutputs(outs0["ValueWeight@GRAD"]);
    }
  }

  if (has_gating) {
    if (outs0.find("GateBias@GRAD") != outs0.end()) {
      outputs[9] = egr::EagerUtils::GetOutputs(outs0["GateBias@GRAD"]);
    }
    if (outs0.find("GateWeight@GRAD") != outs0.end()) {
      outputs[8] = egr::EagerUtils::GetOutputs(outs0["GateWeight@GRAD"]);
    }
  }

  if (NonbatchedBias.defined()) {
    if (outs0.find("NonbatchedBias@GRAD") != outs0.end()) {
      outputs[6] = egr::EagerUtils::GetOutputs(outs0["NonbatchedBias@GRAD"]);
    }
  }

  if (NeedComplexToRealConversion()) HandleComplexGradToRealGrad(&outputs);
  return outputs;
}
