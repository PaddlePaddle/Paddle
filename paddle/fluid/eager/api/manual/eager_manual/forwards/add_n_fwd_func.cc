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

#include "paddle/common/flags.h"
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
#include "paddle/fluid/eager/api/manual/eager_manual/nodes/nodes.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/nan_inf_utils.h"
#include "paddle/fluid/imperative/amp_utils.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

COMMON_DECLARE_bool(check_nan_inf);

paddle::Tensor add_n_ad_func(const std::vector<paddle::Tensor>& x) {
  // Dygraph Record Event
  phi::RecordEvent dygraph_entrance_record_event(
      "add_n dygraph", phi::TracerEventType::Operator, 1);

  // AMP Logic
  if (egr::Controller::Instance().GetAMPLevel() !=
      paddle::imperative::AmpLevel::O0) {
    VLOG(5) << "Check and Prepare For AMP";
    auto op_name = phi::TransToFluidOpName("add_n");
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        amp_tensors_vector = {x};

    auto amp_dst_dtype =
        paddle::imperative::GetAmpDestDtype(op_name, amp_tensors_vector);

    auto NEW_x =
        paddle::imperative::AmpAutoCasts("x", x, amp_dst_dtype, op_name);

    {
      paddle::imperative::AutoCastGuard guard(
          egr::Controller::Instance().GetCurrentAmpAttrs(),
          paddle::imperative::AmpLevel::O0);
      return add_n_ad_func(NEW_x);
    }
  }

  // Get Input AutoGradMeta
  std::vector<egr::AutogradMeta*> x_autograd_meta_vec =
      egr::EagerUtils::nullable_autograd_meta(x);
  std::vector<egr::AutogradMeta*>* x_autograd_meta = &x_autograd_meta_vec;
  // Forward API Call
  VLOG(3) << "Final State Running: "
          << "add_n_ad_func";
  auto api_result = paddle::experimental::add_n(x);

  // Check NaN and Inf if needed
  if (FLAGS_check_nan_inf) {
    egr::CheckTensorHasNanOrInf("add_n", api_result);
  }

  // Get Outputs
  auto& out = api_result;

  // Get Output AutoGradMeta
  egr::AutogradMeta* out_autograd_meta = egr::EagerUtils::autograd_meta(&out);
  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad =
      egr::EagerUtils::ComputeRequireGrad(trace_backward, x_autograd_meta);

  // Check Inplace if needed

  // Node Creation
  if (require_any_grad) {
    phi::RecordEvent node_creation_record_event(
        "add_n node_creation", phi::TracerEventType::OperatorInner, 1);

    egr::EagerUtils::PassStopGradient(false, out_autograd_meta);

    // Node Construction
    auto grad_node = std::shared_ptr<AddNGradNodeFinal>(  // NOLINT
        new AddNGradNodeFinal(1, 1));

    // Set forward's stack
    if (FLAGS_check_nan_inf) {
      grad_node->SetForwardTrace(egr::Controller::Instance().GetPythonStack());
    }

    // SetAttributes if needed

    // Set TensorWrappers for Forward Inputs if needed
    grad_node->SetTensorWrapper_x(x);
    // SetGradOutMeta & SetEdges
    grad_node->SetGradOutMeta(x, 0);
    // SetOutRank & SetHistory & SetGradInMeta & RetainGrad
    if (out_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(out_autograd_meta, 0);
    }
    if (out_autograd_meta) {
      egr::EagerUtils::SetHistory(out_autograd_meta, grad_node);
    }
    grad_node->SetGradInMeta(out, 0);
    // Set TensorWrappers for Forward Outputs if needed
  }

  // Returns
  return out;
}
