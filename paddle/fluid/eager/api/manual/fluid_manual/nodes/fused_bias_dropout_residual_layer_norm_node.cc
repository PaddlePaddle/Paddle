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
fused_bias_dropout_residual_layer_normGradNodeCompat::operator()(
    paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                         egr::kSlotSmallVectorSize>& grads,
    bool create_graph,
    bool is_new_grad) {
  const auto& out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       egr::kSlotSmallVectorSize>
      outputs(5);
  VLOG(3) << "Running Eager Backward Node: "
             "fused_bias_dropout_residual_layer_normGradNodeCompat";
  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       egr::kSlotSmallVectorSize>
      hooked_grads0 = fused_bias_dropout_residual_layer_normGradNodeCompat::
          ApplyGradientHooks(grads);
  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> ins0 =
      {{"BiasDropoutResidualOut",
        egr::EagerUtils::TrySyncToVars(egr::EagerUtils::RecoverTensorWrapper(
            &this->BiasDropoutResidualOut_))},
       {"DropoutMaskOut",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->DropoutMaskOut_))},
       {"LnMean",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->LnMean_))},
       {"LnVariance",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->LnVariance_))},
       {"Residual",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->Residual_))},
       {"X",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->X_))},
       {"Y@GRAD", egr::EagerUtils::TrySyncToVars(hooked_grads0[4])}};

  auto Bias = egr::EagerUtils::RecoverTensorWrapper(&this->Bias_);

  if (Bias.defined()) ins0["Bias"] = egr::EagerUtils::TrySyncToVars(Bias);

  auto LnBias = egr::EagerUtils::RecoverTensorWrapper(&this->LnBias_);
  if (LnBias.defined()) ins0["LnBias"] = egr::EagerUtils::TrySyncToVars(LnBias);
  auto LnScale = egr::EagerUtils::RecoverTensorWrapper(&this->LnScale_);
  if (LnScale.defined())
    ins0["LnScale"] = egr::EagerUtils::TrySyncToVars(LnScale);
  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> outs0;

  if ((!out_metas[0].empty()) && (!(out_metas[0][0].IsStopGradient()))) {
    outs0.insert({"BiasDropoutResidualOut@GRAD",
                  egr::EagerUtils::TrySyncToVars(hooked_grads0[0])});
  }
  if ((!out_metas[1].empty()) && (!(out_metas[1][0].IsStopGradient()))) {
    outs0.insert({"Residual@GRAD",
                  {std::make_shared<egr::EagerVariable>(
                      egr::Controller::Instance().GenerateUniqueName())}});
  }
  if ((!out_metas[0].empty()) && (!(out_metas[0][0].IsStopGradient()))) {
    outs0.insert({"X@GRAD",
                  {std::make_shared<egr::EagerVariable>(
                      egr::Controller::Instance().GenerateUniqueName())}});
  }

  if (Bias.defined() && (!out_metas[2].empty()) &&
      (!out_metas[2][0].IsStopGradient()))
    outs0["Bias@GRAD"] = {std::make_shared<egr::EagerVariable>(
        egr::Controller::Instance().GenerateUniqueName())};
  if (LnBias.defined() && (!out_metas[4].empty()) &&
      (!out_metas[4][0].IsStopGradient()))
    outs0["LnBias@GRAD"] = {std::make_shared<egr::EagerVariable>(
        egr::Controller::Instance().GenerateUniqueName())};
  if (LnScale.defined() && (!out_metas[3].empty()) &&
      (!out_metas[3][0].IsStopGradient()))
    outs0["LnScale@GRAD"] = {std::make_shared<egr::EagerVariable>(
        egr::Controller::Instance().GenerateUniqueName())};
  auto& attrs_map0 = this->attr_map_;
  // Pass the entire attribute map to TraceOp
  // The underlying kernel will pickup whatever attribute they need at runtime

  egr::Controller::Instance().GetCurrentTracer()->TraceOp(
      "fused_bias_dropout_residual_layer_norm_grad",
      ins0,
      outs0,
      attrs_map0,
      egr::Controller::Instance().GetExpectedPlace(),
      &this->default_attr_map_,
      false,
      {});

  if (outs0.find("Bias@GRAD") != outs0.end()) {
    outputs[2] = egr::EagerUtils::GetOutputs(outs0["Bias@GRAD"]);
  }

  if (outs0.find("LnBias@GRAD") != outs0.end()) {
    outputs[4] = egr::EagerUtils::GetOutputs(outs0["LnBias@GRAD"]);
  }

  if (outs0.find("LnScale@GRAD") != outs0.end()) {
    outputs[3] = egr::EagerUtils::GetOutputs(outs0["LnScale@GRAD"]);
  }

  if (outs0.find("Residual@GRAD") != outs0.end()) {
    outputs[1] = egr::EagerUtils::GetOutputs(outs0["Residual@GRAD"]);
  }

  if (outs0.find("X@GRAD") != outs0.end()) {
    outputs[0] = egr::EagerUtils::GetOutputs(outs0["X@GRAD"]);
  }

  if (NeedComplexToRealConversion()) HandleComplexGradToRealGrad(&outputs);
  return outputs;
}
