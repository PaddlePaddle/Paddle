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

paddle::Tensor fused_gemm_epilogue_dygraph_function(
    const paddle::Tensor& X,
    const paddle::Tensor& Y,
    const paddle::Tensor& Bias,
    const paddle::framework::AttributeMap& attr_map) {
  phi::RecordEvent dygraph_entrance_record_event(
      "fused_gemm_epilogue dygraph", phi::TracerEventType::Operator, 1);
  VLOG(3) << "Running Eager Forward Op: fused_gemm_epilogue";
  // Dygraph Forward Pass

  if (egr::Controller::Instance().GetAMPLevel() !=
      paddle::imperative::AmpLevel::O0) {
    VLOG(5) << "Check and Prepare For AMP";

    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        amp_tensors_vector = {{X}, {Y}, {Bias}};

    auto amp_dst_dtype = paddle::imperative::GetAmpDestDtype(
        "fused_gemm_epilogue", amp_tensors_vector);

    auto NEW_X = egr::AmpAutoCast("X", X, amp_dst_dtype, "fused_gemm_epilogue");
    auto NEW_Y = egr::AmpAutoCast("Y", Y, amp_dst_dtype, "fused_gemm_epilogue");
    auto NEW_Bias =
        egr::AmpAutoCast("Bias", Bias, amp_dst_dtype, "fused_gemm_epilogue");

    {
      paddle::imperative::AutoCastGuard guard(
          egr::Controller::Instance().GetCurrentAmpAttrs(),
          paddle::imperative::AmpLevel::O0);
      return fused_gemm_epilogue_dygraph_function(
          NEW_X, NEW_Y, NEW_Bias, attr_map);
    }
  }

  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> ins =
      {{"X", egr::EagerUtils::TrySyncToVars(X)},
       {"Y", egr::EagerUtils::TrySyncToVars(Y)},
       {"Bias", egr::EagerUtils::TrySyncToVars(Bias)}};

  std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>> outs =
      {{"Out",
        {std::make_shared<egr::EagerVariable>(
            egr::Controller::Instance().GenerateUniqueName())}}};

  // Prepare Autograd Meta
  egr::AutogradMeta* p_autograd_X = egr::EagerUtils::nullable_autograd_meta(X);
  egr::AutogradMeta* p_autograd_Y = egr::EagerUtils::nullable_autograd_meta(Y);
  egr::AutogradMeta* p_autograd_Bias =
      egr::EagerUtils::nullable_autograd_meta(Bias);

  bool trace_backward = egr::Controller::Instance().HasGrad();

  bool require_any_grad = egr::EagerUtils::ComputeRequireGrad(
      trace_backward, p_autograd_X, p_autograd_Y, p_autograd_Bias);

  paddle::framework::AttributeMap attrs = attr_map;
  paddle::framework::AttributeMap default_attrs;
  egr::Controller::Instance().GetCurrentTracer()->TraceOp(
      "fused_gemm_epilogue",
      ins,
      outs,
      attrs,
      egr::Controller::Instance().GetExpectedPlace(),
      &default_attrs,
      true,
      {});

  paddle::Tensor Out;
  egr::EagerUtils::GetOutput(outs["Out"][0], &Out);

  {
    phi::RecordEvent node_creation_record_event(
        "fused_gemm_epilogue node_creation",
        phi::TracerEventType::OperatorInner,
        1);
    egr::AutogradMeta* p_autograd_Out = egr::EagerUtils::autograd_meta(&Out);
    if (require_any_grad) {
      VLOG(6) << " Construct Grad for fused_gemm_epilogue ";
      egr::EagerUtils::PassStopGradient(false, p_autograd_Out);
      // Create GradOpNode
      auto grad_node =
          std::shared_ptr<fused_gemm_epilogueGradNodeCompat>(  // NOLINT
              new fused_gemm_epilogueGradNodeCompat(1, 3));

      // Set Attributes
      grad_node->SetAttrMap(std::move(attrs));
      grad_node->SetDefaultAttrMap(std::move(default_attrs));

      // Set Tensor Wrappers
      grad_node->SetTensorWrapper_X(X);
      grad_node->SetTensorWrapper_Y(Y);

      grad_node->SetGradOutMeta(X, 0);
      grad_node->SetGradOutMeta(Y, 1);
      grad_node->SetGradOutMeta(Bias, 2);
      egr::EagerUtils::SetOutRankWithSlot(p_autograd_Out, 0);
      egr::EagerUtils::SetHistory(p_autograd_Out, grad_node);
      grad_node->SetGradInMeta(Out, 0);
    }
  }

  return Out;
}
