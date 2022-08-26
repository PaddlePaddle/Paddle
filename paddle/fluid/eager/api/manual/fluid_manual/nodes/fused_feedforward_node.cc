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
fused_feedforwardGradNodeCompat::operator()(
    paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                         egr::kSlotSmallVectorSize>& grads,
    bool create_graph,
    bool is_new_grad) {
  VLOG(3) << "Running Eager Backward Node: fused_feedforwardGradNodeCompat";
  const auto& out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       egr::kSlotSmallVectorSize>
      outputs(11);

  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       egr::kSlotSmallVectorSize>
      hooked_grads0 =
          fused_feedforwardGradNodeCompat::ApplyGradientHooks(grads);

  bool pre_layer_norm = false;
  if (attr_map_.count("pre_layer_norm")) {
    pre_layer_norm = PADDLE_GET_CONST(bool, attr_map_.at("pre_layer_norm"));
  }

  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> ins0 =
      {{"Dropout1Mask",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->Dropout1Mask_))},
       {"Dropout1Out",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->Dropout1Out_))},
       {"Dropout2Mask",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->Dropout2Mask_))},
       {"Dropout2Out",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->Dropout2Out_))},
       {"Linear1Out",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->Linear1Out_))},
       {"Linear1Weight",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->Linear1Weight_))},
       {"Linear2Weight",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->Linear2Weight_))},
       {"Out@GRAD", egr::EagerUtils::TrySyncToVars(hooked_grads0[0])},
       {"X",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->X_))}};

  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> outs0;

  auto Linear1Bias = egr::EagerUtils::RecoverTensorWrapper(&this->Linear1Bias_);
  if (Linear1Bias.defined())
    ins0["Linear1Bias"] = egr::EagerUtils::TrySyncToVars(Linear1Bias);

  if ((!out_metas[3].empty()) && (!(out_metas[3][0].IsStopGradient()))) {
    outs0.insert({"Linear1Weight@GRAD",
                  {std::make_shared<egr::EagerVariable>(
                      egr::Controller::Instance().GenerateUniqueName())}});
  }
  if ((!out_metas[5].empty()) && (!(out_metas[5][0].IsStopGradient()))) {
    outs0.insert({"Linear2Weight@GRAD",
                  {std::make_shared<egr::EagerVariable>(
                      egr::Controller::Instance().GenerateUniqueName())}});
  }
  if ((!out_metas[0].empty()) && (!(out_metas[0][0].IsStopGradient()))) {
    outs0.insert({"X@GRAD",
                  {std::make_shared<egr::EagerVariable>(
                      egr::Controller::Instance().GenerateUniqueName())}});
  }
  if (Linear1Bias.defined() && (!out_metas[4].empty()) &&
      (!out_metas[4][0].IsStopGradient()))
    outs0["Linear1Bias@GRAD"] = {std::make_shared<egr::EagerVariable>(
        egr::Controller::Instance().GenerateUniqueName())};

  if (pre_layer_norm) {
    auto Ln1Scale = egr::EagerUtils::RecoverTensorWrapper(&this->Ln1Scale_);
    if (Ln1Scale.defined())
      ins0["Ln1Scale"] = egr::EagerUtils::TrySyncToVars(Ln1Scale);
    auto Ln1Bias = egr::EagerUtils::RecoverTensorWrapper(&this->Ln1Bias_);
    if (Ln1Bias.defined())
      ins0["Ln1Bias"] = egr::EagerUtils::TrySyncToVars(Ln1Bias);
    auto Ln1Out = egr::EagerUtils::RecoverTensorWrapper(&this->Ln1Out_);
    if (Ln1Out.defined())
      ins0["Ln1Out"] = egr::EagerUtils::TrySyncToVars(Ln1Out);
    auto Ln1Mean = egr::EagerUtils::RecoverTensorWrapper(&this->Ln1Mean_);
    if (Ln1Mean.defined())
      ins0["Ln1Mean"] = egr::EagerUtils::TrySyncToVars(Ln1Mean);
    auto Ln1Variance =
        egr::EagerUtils::RecoverTensorWrapper(&this->Ln1Variance_);
    if (Ln1Variance.defined())
      ins0["Ln1Variance"] = egr::EagerUtils::TrySyncToVars(Ln1Variance);
    if (Ln1Scale.defined() && (!out_metas[7].empty()) &&
        (!out_metas[7][0].IsStopGradient()))
      outs0["Ln1Scale@GRAD"] = {std::make_shared<egr::EagerVariable>(
          egr::Controller::Instance().GenerateUniqueName())};
    if (Ln1Bias.defined() && (!out_metas[8].empty()) &&
        (!out_metas[8][0].IsStopGradient()))
      outs0["Ln1Bias@GRAD"] = {std::make_shared<egr::EagerVariable>(
          egr::Controller::Instance().GenerateUniqueName())};

  } else {
    auto Ln2Scale = egr::EagerUtils::RecoverTensorWrapper(&this->Ln2Scale_);
    if (Ln2Scale.defined())
      ins0["Ln2Scale"] = egr::EagerUtils::TrySyncToVars(Ln2Scale);
    auto Ln2Bias = egr::EagerUtils::RecoverTensorWrapper(&this->Ln2Bias_);
    if (Ln2Bias.defined())
      ins0["Ln2Bias"] = egr::EagerUtils::TrySyncToVars(Ln2Bias);
    auto Ln2Mean = egr::EagerUtils::RecoverTensorWrapper(&this->Ln2Mean_);
    if (Ln2Mean.defined())
      ins0["Ln2Mean"] = egr::EagerUtils::TrySyncToVars(Ln2Mean);
    auto Ln2Variance =
        egr::EagerUtils::RecoverTensorWrapper(&this->Ln2Variance_);
    if (Ln2Variance.defined())
      ins0["Ln2Variance"] = egr::EagerUtils::TrySyncToVars(Ln2Variance);
    if (Ln2Scale.defined() && (!out_metas[9].empty()) &&
        (!out_metas[9][0].IsStopGradient()))
      outs0["Ln2Scale@GRAD"] = {std::make_shared<egr::EagerVariable>(
          egr::Controller::Instance().GenerateUniqueName())};
    if (Ln2Bias.defined() && (!out_metas[10].empty()) &&
        (!out_metas[10][0].IsStopGradient()))
      outs0["Ln2Bias@GRAD"] = {std::make_shared<egr::EagerVariable>(
          egr::Controller::Instance().GenerateUniqueName())};
  }

  auto Linear2Bias = egr::EagerUtils::RecoverTensorWrapper(&this->Linear2Bias_);
  if (Linear2Bias.defined()) {
    ins0["Linear2Bias"] = egr::EagerUtils::TrySyncToVars(Linear2Bias);
    if ((!out_metas[6].empty()) && (!out_metas[6][0].IsStopGradient()))
      outs0["Linear2Bias@GRAD"] = {std::make_shared<egr::EagerVariable>(
          egr::Controller::Instance().GenerateUniqueName())};
  }

  auto& attrs_map0 = this->attr_map_;
  // Pass the entire attribute map to TraceOp
  // The underlying kernel will pickup whatever attribute they need at runtime
  egr::Controller::Instance().GetCurrentTracer()->TraceOp(
      "fused_feedforward_grad",
      ins0,
      outs0,
      attrs_map0,
      egr::Controller::Instance().GetExpectedPlace(),
      &this->default_attr_map_,
      false,
      {});

  if (outs0.find("Linear1Weight@GRAD") != outs0.end()) {
    outputs[3] = egr::EagerUtils::GetOutputs(outs0["Linear1Weight@GRAD"]);
  }
  if (outs0.find("Linear2Weight@GRAD") != outs0.end()) {
    outputs[5] = egr::EagerUtils::GetOutputs(outs0["Linear2Weight@GRAD"]);
  }
  if (outs0.find("X@GRAD") != outs0.end()) {
    outputs[0] = egr::EagerUtils::GetOutputs(outs0["X@GRAD"]);
  }
  if (outs0.find("Linear1Bias@GRAD") != outs0.end()) {
    outputs[4] = egr::EagerUtils::GetOutputs(outs0["Linear1Bias@GRAD"]);
  }

  if (pre_layer_norm) {
    if (outs0.find("Ln1Scale@GRAD") != outs0.end()) {
      outputs[7] = egr::EagerUtils::GetOutputs(outs0["Ln1Scale@GRAD"]);
    }
    if (outs0.find("Ln1Bias@GRAD") != outs0.end()) {
      outputs[8] = egr::EagerUtils::GetOutputs(outs0["Ln1Bias@GRAD"]);
    }

  } else {
    if (outs0.find("Ln2Bias@GRAD") != outs0.end()) {
      outputs[10] = egr::EagerUtils::GetOutputs(outs0["Ln2Bias@GRAD"]);
    }
    if (outs0.find("Ln2Scale@GRAD") != outs0.end()) {
      outputs[9] = egr::EagerUtils::GetOutputs(outs0["Ln2Scale@GRAD"]);
    }
  }

  if (Linear2Bias.defined()) {
    if (outs0.find("Linear2Bias@GRAD") != outs0.end()) {
      outputs[6] = egr::EagerUtils::GetOutputs(outs0["Linear2Bias@GRAD"]);
    }
  }

  if (NeedComplexToRealConversion()) HandleComplexGradToRealGrad(&outputs);
  return outputs;
}
