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
fused_gemm_epilogueGradNodeCompat::operator()(
    paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                         egr::kSlotSmallVectorSize>& grads,
    bool create_graph,
    bool is_new_grad) {
  const auto& out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       egr::kSlotSmallVectorSize>
      outputs(3);
  VLOG(3) << "Running Eager Backward Node: fused_gemm_epilogueGradNodeCompat";
  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       egr::kSlotSmallVectorSize>
      hooked_grads0 =
          fused_gemm_epilogueGradNodeCompat::ApplyGradientHooks(grads);
  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> ins0 =
      {{"DOut", egr::EagerUtils::TrySyncToVars(hooked_grads0[0])},
       {"X",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->X_))},
       {"Y",
        egr::EagerUtils::TrySyncToVars(
            egr::EagerUtils::RecoverTensorWrapper(&this->Y_))}};
  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> outs0;
  if ((!out_metas[2].empty()) && (!(out_metas[2][0].IsStopGradient()))) {
    outs0.insert({"DBias",
                  {std::make_shared<egr::EagerVariable>(
                      egr::Controller::Instance().GenerateUniqueName())}});
  }
  if ((!out_metas[0].empty()) && (!(out_metas[0][0].IsStopGradient()))) {
    outs0.insert({"DX",
                  {std::make_shared<egr::EagerVariable>(
                      egr::Controller::Instance().GenerateUniqueName())}});
  }
  if ((!out_metas[1].empty()) && (!(out_metas[1][0].IsStopGradient()))) {
    outs0.insert({"DY",
                  {std::make_shared<egr::EagerVariable>(
                      egr::Controller::Instance().GenerateUniqueName())}});
  }

  auto& attrs_map0 = this->attr_map_;
  // Pass the entire attribute map to TraceOp
  // The underlying kernel will pickup whatever attribute they need at runtime
  egr::Controller::Instance().GetCurrentTracer()->TraceOp(
      "fused_gemm_epilogue_grad",
      ins0,
      outs0,
      attrs_map0,
      egr::Controller::Instance().GetExpectedPlace(),
      &this->default_attr_map_,
      true,
      {});
  if (outs0.find("DBias") != outs0.end()) {
    outputs[2] = egr::EagerUtils::GetOutputs(outs0["DBias"]);
  }
  if (outs0.find("DX") != outs0.end()) {
    outputs[0] = egr::EagerUtils::GetOutputs(outs0["DX"]);
  }
  if (outs0.find("DY") != outs0.end()) {
    outputs[1] = egr::EagerUtils::GetOutputs(outs0["DY"]);
  }

  if (NeedComplexToRealConversion()) HandleComplexGradToRealGrad(&outputs);
  return outputs;
}
