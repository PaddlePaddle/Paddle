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
fused_attentionGradNodeCompat::operator()(
    paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                         egr::kSlotSmallVectorSize>& grads,
    bool create_graph,
    bool is_new_grad) {
  VLOG(3) << "Running Eager Backward Node: fused_attentionGradNodeCompat";
  const auto& out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       egr::kSlotSmallVectorSize>
      outputs(23);
  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       egr::kSlotSmallVectorSize>
      hooked_grads0 = fused_attentionGradNodeCompat::ApplyGradientHooks(grads);

  bool pre_layer_norm = false;
  if (attr_map_.count("pre_layer_norm")) {
    pre_layer_norm = PADDLE_GET_CONST(bool, attr_map_.at("pre_layer_norm"));
  }

  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> ins0 =
      {{"AttnDropoutMaskOut",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->AttnDropoutMaskOut_))},
       {"AttnDropoutOut",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->AttnDropoutOut_))},
       {"DropoutMaskOut",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->DropoutMaskOut_))},
       {"FMHAOut",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->FMHAOut_))},
       {"OutLinearOut",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->OutLinearOut_))},
       {"OutLinearW",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->OutLinearW_))},
       {"QKOut",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->QKOut_))},
       {"QKTVOut",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->QKTVOut_))},
       {"QKVOut",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->QKVOut_))},
       {"QKVW",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->QKVW_))},
       {"SoftmaxOut",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->SoftmaxOut_))},
       {"TransposeOut2",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->TransposeOut2_))},
       {"X",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->X_))},
       {"Y@GRAD", egr::EagerUtils::TrySyncToVars(hooked_grads0[19])}};
  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> outs0;

  if ((!out_metas[7].empty()) && (!(out_metas[7][0].IsStopGradient()))) {
    outs0.insert({"OutLinearW@GRAD",
                  {std::make_shared<egr::EagerVariable>(
                      egr::Controller::Instance().GenerateUniqueName())}});
  }
  if ((!out_metas[3].empty()) && (!(out_metas[3][0].IsStopGradient()))) {
    outs0.insert({"QKVW@GRAD",
                  {std::make_shared<egr::EagerVariable>(
                      egr::Controller::Instance().GenerateUniqueName())}});
  }
  if ((!out_metas[0].empty()) && (!(out_metas[0][0].IsStopGradient()))) {
    outs0.insert({"X@GRAD",
                  {std::make_shared<egr::EagerVariable>(
                      egr::Controller::Instance().GenerateUniqueName())}});
  }

  auto QKVOut = egr::EagerUtils::RecoverTensorWrapper(&this->QKVOut_);
  if (QKVOut.defined() && (!out_metas[15].empty()) &&
      (!out_metas[15][0].IsStopGradient()))
    outs0["QKVOut@GRAD"] = {std::make_shared<egr::EagerVariable>(
        egr::Controller::Instance().GenerateUniqueName())};
  auto QKTVOut = egr::EagerUtils::RecoverTensorWrapper(&this->QKTVOut_);
  if (QKTVOut.defined() && (!out_metas[16].empty()) &&
      (!out_metas[16][0].IsStopGradient()))
    outs0["QKTVOut@GRAD"] = {std::make_shared<egr::EagerVariable>(
        egr::Controller::Instance().GenerateUniqueName())};
  auto TransposeOut2 =
      egr::EagerUtils::RecoverTensorWrapper(&this->TransposeOut2_);
  if (TransposeOut2.defined() && (!out_metas[17].empty()) &&
      (!out_metas[17][0].IsStopGradient()))
    outs0["TransposeOut2@GRAD"] = {std::make_shared<egr::EagerVariable>(
        egr::Controller::Instance().GenerateUniqueName())};
  auto QKOut = egr::EagerUtils::RecoverTensorWrapper(&this->QKOut_);
  if (QKOut.defined() && (!out_metas[18].empty()) &&
      (!out_metas[18][0].IsStopGradient()))
    outs0["QKOut@GRAD"] = {std::make_shared<egr::EagerVariable>(
        egr::Controller::Instance().GenerateUniqueName())};
  auto SoftmaxOut = egr::EagerUtils::RecoverTensorWrapper(&this->SoftmaxOut_);
  if (SoftmaxOut.defined() && (!out_metas[19].empty()) &&
      (!out_metas[19][0].IsStopGradient()))
    outs0["SoftmaxOut@GRAD"] = {std::make_shared<egr::EagerVariable>(
        egr::Controller::Instance().GenerateUniqueName())};
  auto AttnDropoutOut =
      egr::EagerUtils::RecoverTensorWrapper(&this->AttnDropoutOut_);
  if (AttnDropoutOut.defined() && (!out_metas[20].empty()) &&
      (!out_metas[20][0].IsStopGradient()))
    outs0["AttnDropoutOut@GRAD"] = {std::make_shared<egr::EagerVariable>(
        egr::Controller::Instance().GenerateUniqueName())};
  auto FMHAOut = egr::EagerUtils::RecoverTensorWrapper(&this->FMHAOut_);
  if (FMHAOut.defined() && (!out_metas[21].empty()) &&
      (!out_metas[21][0].IsStopGradient()))
    outs0["FMHAOut@GRAD"] = {std::make_shared<egr::EagerVariable>(
        egr::Controller::Instance().GenerateUniqueName())};
  auto OutLinearOut =
      egr::EagerUtils::RecoverTensorWrapper(&this->OutLinearOut_);
  if (OutLinearOut.defined() && (!out_metas[22].empty()) &&
      (!out_metas[22][0].IsStopGradient()))
    outs0["OutLinearOut@GRAD"] = {std::make_shared<egr::EagerVariable>(
        egr::Controller::Instance().GenerateUniqueName())};

  auto QKVBias = egr::EagerUtils::RecoverTensorWrapper(&this->QKVBias_);
  if (QKVBias.defined()) {
    ins0["QKVBias"] = egr::EagerUtils::TrySyncToVars(QKVBias);
    auto QKVBiasOut = egr::EagerUtils::RecoverTensorWrapper(&this->QKVBiasOut_);
    ins0["QKVBiasOut"] = egr::EagerUtils::TrySyncToVars(QKVBiasOut);
    if (QKVBias.defined() && (!out_metas[4].empty()) &&
        (!out_metas[4][0].IsStopGradient()))
      outs0["QKVBias@GRAD"] = {std::make_shared<egr::EagerVariable>(
          egr::Controller::Instance().GenerateUniqueName())};
    if (QKVBiasOut.defined() && (!out_metas[11].empty()) &&
        (!out_metas[11][0].IsStopGradient()))
      outs0["QKVBiasOut@GRAD"] = {std::make_shared<egr::EagerVariable>(
          egr::Controller::Instance().GenerateUniqueName())};
  }

  auto SrcMask = egr::EagerUtils::RecoverTensorWrapper(&this->SrcMask_);
  if (SrcMask.defined()) {
    ins0["SrcMask"] = egr::EagerUtils::TrySyncToVars(SrcMask);
    auto SrcMaskOut = egr::EagerUtils::RecoverTensorWrapper(&this->SrcMaskOut_);
    ins0["SrcMaskOut"] = egr::EagerUtils::TrySyncToVars(SrcMaskOut);
    if (SrcMaskOut.defined() && (!out_metas[12].empty()) &&
        (!out_metas[12][0].IsStopGradient()))
      outs0["SrcMaskOut@GRAD"] = {std::make_shared<egr::EagerVariable>(
          egr::Controller::Instance().GenerateUniqueName())};
  }

  auto OutLinearBias =
      egr::EagerUtils::RecoverTensorWrapper(&this->OutLinearBias_);
  if (OutLinearBias.defined()) {
    ins0["OutLinearBias"] = egr::EagerUtils::TrySyncToVars(OutLinearBias);
    if (OutLinearBias.defined() && (!out_metas[8].empty()) &&
        (!out_metas[8][0].IsStopGradient()))
      outs0["OutLinearBias@GRAD"] = {std::make_shared<egr::EagerVariable>(
          egr::Controller::Instance().GenerateUniqueName())};
  }

  if (pre_layer_norm) {
    auto LnScale = egr::EagerUtils::RecoverTensorWrapper(&this->LnScale_);
    if (LnScale.defined()) {
      ins0["LnScale"] = egr::EagerUtils::TrySyncToVars(LnScale);
      if (LnScale.defined() && (!out_metas[1].empty()) &&
          (!out_metas[1][0].IsStopGradient()))
        outs0["LnScale@GRAD"] = {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())};
    }

    auto LnBias = egr::EagerUtils::RecoverTensorWrapper(&this->LnBias_);
    if (LnBias.defined()) {
      ins0["LnBias"] = egr::EagerUtils::TrySyncToVars(LnBias);
      if (LnBias.defined() && (!out_metas[2].empty()) &&
          (!out_metas[2][0].IsStopGradient()))
        outs0["LnBias@GRAD"] = {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())};
    }

    auto LnOut = egr::EagerUtils::RecoverTensorWrapper(&this->LnOut_);
    if (LnOut.defined()) {
      ins0["LnOut"] = egr::EagerUtils::TrySyncToVars(LnOut);
      if (LnOut.defined() && (!out_metas[13].empty()) &&
          (!out_metas[13][0].IsStopGradient()))
        outs0["LnOut@GRAD"] = {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())};
    }

    auto LnMean = egr::EagerUtils::RecoverTensorWrapper(&this->LnMean_);
    if (LnMean.defined()) {
      ins0["LnMean"] = egr::EagerUtils::TrySyncToVars(LnMean);
    }

    auto LnVariance = egr::EagerUtils::RecoverTensorWrapper(&this->LnVariance_);
    if (LnVariance.defined()) {
      ins0["LnVariance"] = egr::EagerUtils::TrySyncToVars(LnVariance);
    }
  } else {
    auto Ln2Scale = egr::EagerUtils::RecoverTensorWrapper(&this->Ln2Scale_);
    if (Ln2Scale.defined()) {
      ins0["Ln2Scale"] = egr::EagerUtils::TrySyncToVars(Ln2Scale);
      if (Ln2Scale.defined() && (!out_metas[9].empty()) &&
          (!out_metas[9][0].IsStopGradient()))
        outs0["Ln2Scale@GRAD"] = {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())};
    }

    auto Ln2Bias = egr::EagerUtils::RecoverTensorWrapper(&this->Ln2Bias_);
    if (Ln2Bias.defined()) {
      ins0["Ln2Bias"] = egr::EagerUtils::TrySyncToVars(Ln2Bias);
      if (Ln2Bias.defined() && (!out_metas[10].empty()) &&
          (!out_metas[10][0].IsStopGradient()))
        outs0["Ln2Bias@GRAD"] = {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())};
    }
    auto BiasDropoutResidualOut =
        egr::EagerUtils::RecoverTensorWrapper(&this->BiasDropoutResidualOut_);
    auto Ln2Mean = egr::EagerUtils::RecoverTensorWrapper(&this->Ln2Mean_);
    auto Ln2Variance =
        egr::EagerUtils::RecoverTensorWrapper(&this->Ln2Variance_);
    ins0["BiasDropoutResidualOut"] =
        egr::EagerUtils::TrySyncToVars(BiasDropoutResidualOut);
    ins0["Ln2Mean"] = egr::EagerUtils::TrySyncToVars(Ln2Mean);
    ins0["Ln2Variance"] = egr::EagerUtils::TrySyncToVars(Ln2Variance);
    if (BiasDropoutResidualOut.defined() && (!out_metas[14].empty()) &&
        (!out_metas[14][0].IsStopGradient()))
      outs0["BiasDropoutResidualOut@GRAD"] = {
          std::make_shared<egr::EagerVariable>(
              egr::Controller::Instance().GenerateUniqueName())};
  }

  auto& attrs_map0 = this->attr_map_;
  // Pass the entire attribute map to TraceOp
  // The underlying kernel will pickup whatever attribute they need at runtime
  egr::Controller::Instance().GetCurrentTracer()->TraceOp(
      "fused_attention_grad",
      ins0,
      outs0,
      attrs_map0,
      egr::Controller::Instance().GetExpectedPlace(),
      &this->default_attr_map_,
      false,
      {});

  if (outs0.find("OutLinearW@GRAD") != outs0.end()) {
    outputs[7] = egr::EagerUtils::GetOutputs(outs0["OutLinearW@GRAD"]);
  }
  if (outs0.find("QKVW@GRAD") != outs0.end()) {
    outputs[3] = egr::EagerUtils::GetOutputs(outs0["QKVW@GRAD"]);
  }
  if (outs0.find("X@GRAD") != outs0.end()) {
    outputs[0] = egr::EagerUtils::GetOutputs(outs0["X@GRAD"]);
  }

  if (outs0.find("QKVOut@GRAD") != outs0.end()) {
    outputs[15] = egr::EagerUtils::GetOutputs(outs0["QKVOut@GRAD"]);
  }
  if (outs0.find("QKTVOut@GRAD") != outs0.end()) {
    outputs[16] = egr::EagerUtils::GetOutputs(outs0["QKTVOut@GRAD"]);
  }
  if (outs0.find("TransposeOut2@GRAD") != outs0.end()) {
    outputs[17] = egr::EagerUtils::GetOutputs(outs0["TransposeOut2@GRAD"]);
  }
  if (outs0.find("QKOut@GRAD") != outs0.end()) {
    outputs[18] = egr::EagerUtils::GetOutputs(outs0["QKOut@GRAD"]);
  }
  if (outs0.find("SoftmaxOut@GRAD") != outs0.end()) {
    outputs[19] = egr::EagerUtils::GetOutputs(outs0["SoftmaxOut@GRAD"]);
  }
  if (outs0.find("AttnDropoutOut@GRAD") != outs0.end()) {
    outputs[20] = egr::EagerUtils::GetOutputs(outs0["AttnDropoutOut@GRAD"]);
  }
  if (outs0.find("FMHAOut@GRAD") != outs0.end()) {
    outputs[21] = egr::EagerUtils::GetOutputs(outs0["FMHAOut@GRAD"]);
  }
  if (outs0.find("OutLinearOut@GRAD") != outs0.end()) {
    outputs[22] = egr::EagerUtils::GetOutputs(outs0["OutLinearOut@GRAD"]);
  }

  if (QKVBias.defined()) {
    if (outs0.find("QKVBias@GRAD") != outs0.end()) {
      outputs[4] = egr::EagerUtils::GetOutputs(outs0["QKVBias@GRAD"]);
    }
    if (outs0.find("QKVBiasOut@GRAD") != outs0.end()) {
      outputs[11] = egr::EagerUtils::GetOutputs(outs0["QKVBiasOut@GRAD"]);
    }
  }

  if (SrcMask.defined()) {
    if (outs0.find("SrcMaskOut@GRAD") != outs0.end()) {
      outputs[12] = egr::EagerUtils::GetOutputs(outs0["SrcMaskOut@GRAD"]);
    }
  }

  if (OutLinearBias.defined()) {
    if (outs0.find("OutLinearBias@GRAD") != outs0.end()) {
      outputs[8] = egr::EagerUtils::GetOutputs(outs0["OutLinearBias@GRAD"]);
    }
  }

  if (pre_layer_norm) {
    auto LnScale = egr::EagerUtils::RecoverTensorWrapper(&this->LnScale_);
    if (LnScale.defined()) {
      if (outs0.find("LnScale@GRAD") != outs0.end()) {
        outputs[1] = egr::EagerUtils::GetOutputs(outs0["LnScale@GRAD"]);
      }
    }

    auto LnBias = egr::EagerUtils::RecoverTensorWrapper(&this->LnBias_);
    if (LnBias.defined()) {
      if (outs0.find("LnBias@GRAD") != outs0.end()) {
        outputs[2] = egr::EagerUtils::GetOutputs(outs0["LnBias@GRAD"]);
      }
    }

    auto LnOut = egr::EagerUtils::RecoverTensorWrapper(&this->LnOut_);
    if (LnOut.defined()) {
      if (outs0.find("LnOut@GRAD") != outs0.end()) {
        outputs[13] = egr::EagerUtils::GetOutputs(outs0["LnOut@GRAD"]);
      }
    }
  } else {
    auto Ln2Scale = egr::EagerUtils::RecoverTensorWrapper(&this->Ln2Scale_);
    if (Ln2Scale.defined()) {
      if (outs0.find("Ln2Scale@GRAD") != outs0.end()) {
        outputs[9] = egr::EagerUtils::GetOutputs(outs0["Ln2Scale@GRAD"]);
      }
    }

    auto Ln2Bias = egr::EagerUtils::RecoverTensorWrapper(&this->Ln2Bias_);
    if (Ln2Bias.defined()) {
      if (outs0.find("Ln2Bias@GRAD") != outs0.end()) {
        outputs[10] = egr::EagerUtils::GetOutputs(outs0["Ln2Bias@GRAD"]);
      }
    }
    if (outs0.find("BiasDropoutResidualOut@GRAD") != outs0.end()) {
      outputs[14] =
          egr::EagerUtils::GetOutputs(outs0["BiasDropoutResidualOut@GRAD"]);
    }
  }

  if (NeedComplexToRealConversion()) HandleComplexGradToRealGrad(&outputs);
  return outputs;
}
