// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
#include "paddle/fluid/eager/api/manual/eager_manual/nodes/nodes.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

paddle::Tensor dtensor_to_local_ad_function(const paddle::Tensor& input) {
#ifdef PADDLE_WITH_DISTRIBUTE
  VLOG(3) << "Running AD API: "
          << "dtensor_to_local dygraph";
  // Dygraph Record Event
  phi::RecordEvent dygraph_entrance_record_event(
      "dtensor_to_local dygraph", phi::TracerEventType::Communication, 1);

  // Get Input AutoGradMeta
  egr::AutogradMeta* input_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(input);
  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad =
      egr::EagerUtils::ComputeRequireGrad(trace_backward, input_autograd_meta);

  // Node Declaration
  std::shared_ptr<DtensorToLocalGradNode> grad_node;

  // Set grad_node before API Call
  if (require_any_grad) {
    phi::RecordEvent node_creation_record_event(
        "dtensor_to_local node_creation",
        phi::TracerEventType::Communication,
        1);

    // Node Construction
    grad_node = std::shared_ptr<DtensorToLocalGradNode>(
        new DtensorToLocalGradNode(1, 1));  // NOLINT

    // Set TensorWrappers for Forward Inputs if needed
    grad_node->SetTensorWrapperNoNeedBuffer_Input(input);
  }

  // Forward API Call
  // dtensor_to_local_func(input, api_result, dist_attr);
  PADDLE_ENFORCE_EQ(
      input.initialized(),
      true,
      phi::errors::InvalidArgument("Input tensor must be initialized."));

  paddle::Tensor api_result;
  if (input.is_dist_tensor()) {
    std::shared_ptr<phi::TensorBase> tensor_base = input.impl();

    auto dist_tensor =
        std::static_pointer_cast<phi::distributed::DistTensor>(tensor_base);

    std::shared_ptr<phi::DenseTensor> local_dense = dist_tensor->shared_value();

    PADDLE_ENFORCE_NE(local_dense,
                      nullptr,
                      phi::errors::InvalidArgument(
                          "The local DenseTensor inside DistTensor is null."));

    PADDLE_ENFORCE_EQ(
        local_dense->initialized(),
        true,
        phi::errors::PreconditionNotMet(
            "The local DenseTensor inside DistTensor is not initialized."));

    api_result = paddle::Tensor(local_dense);
  } else {
    api_result = input;
  }

  // Get Outputs
  auto& out = api_result;

  // Get Output AutoGradMeta
  egr::AutogradMeta* out_autograd_meta = egr::EagerUtils::autograd_meta(&out);

  // Set grad_node after API call
  if (require_any_grad) {
    egr::EagerUtils::PassStopGradient(false, out_autograd_meta);

    // SetGradOutMeta & SetEdges
    grad_node->SetGradOutMeta(input, 0);
    // SetOutRank & SetHistory & SetGradInMeta
    if (out_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(out_autograd_meta, 0);
      egr::EagerUtils::SetHistory(out_autograd_meta, grad_node);
    }
    grad_node->SetGradInMeta(out, 0);
  }

  return out;
#else
  PADDLE_THROW(common::errors::Unavailable(
      "dtensor_to_local is not supported in this version of Paddle. Try to "
      "recompile it "
      "with WITH_DISTRIBUTE=ON and reinstall this package."));
  return paddle::Tensor();
#endif
}
