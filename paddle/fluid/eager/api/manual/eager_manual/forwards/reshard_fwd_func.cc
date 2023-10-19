// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/eager/amp_utils.h"
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
#include "paddle/fluid/eager/api/manual/eager_manual/nodes/nodes.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_function.h"
#include "paddle/phi/core/enforce.h"

paddle::Tensor reshard_ad_function(
    const paddle::Tensor& input,
    const phi::distributed::TensorDistAttr dist_attr) {
  VLOG(3) << "Running AD API: "
          << "reshard dygraph";
  // Dygraph Record Event
  paddle::platform::RecordEvent dygraph_entrance_record_event(
      "reshard dygraph", paddle::platform::TracerEventType::Communication, 1);

  // Get Input AutoGradMeta
  egr::AutogradMeta* input_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(input);
  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad =
      egr::EagerUtils::ComputeRequireGrad(trace_backward, input_autograd_meta);

  // Node Declaration
  std::shared_ptr<ReshardGradNode> grad_node;

  // Set grad_node before API Call
  if (require_any_grad) {
    paddle::platform::RecordEvent node_creation_record_event(
        "reshard node_creation",
        paddle::platform::TracerEventType::Communication,
        1);

    // Node Construction
    grad_node =
        std::shared_ptr<ReshardGradNode>(new ReshardGradNode(1, 1));  // NOLINT

    // Set TensorWrappers for Forward Inputs if needed
    grad_node->SetTensorWrapperNoNeedBufferInput(input);
  }

  // Forward API Call
  // reshard_func(input, api_result, dist_attr);
  auto dev_ctx = phi::DeviceContextPool::Instance().Get(input.place());
  std::shared_ptr<phi::distributed::DistTensor> dist_out_ptr = nullptr;
  if (phi::distributed::DistTensor::classof(input.impl().get())) {
    auto tensor_in = input.impl();
    if (tensor_in) {
      phi::distributed::DistTensor* dist_tensor =
          static_cast<phi::distributed::DistTensor*>(tensor_in.get());
      if (dist_tensor->dist_attr() != dist_attr) {
        VLOG(6) << "reshard func, reshard tensor from "
                << dist_tensor->dist_attr() << " to " << dist_attr;
        auto* func = phi::distributed::ChooseProperReshardFunction(*dist_tensor,
                                                                   dist_attr);
        dist_out_ptr = func->Eval(dev_ctx, *dist_tensor, dist_attr);
      } else {
        dist_out_ptr =
            std::static_pointer_cast<phi::distributed::DistTensor>(tensor_in);
      }
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "The input tensor of shard function should be "
        "``phi::distributed::DistTensor``. "
        "However it's %s",
        typeid(input.impl().get()).name()));
  }
  auto api_result = paddle::Tensor(dist_out_ptr);

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
}
