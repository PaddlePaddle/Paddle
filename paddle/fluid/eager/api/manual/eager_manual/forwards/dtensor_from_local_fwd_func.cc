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
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

paddle::Tensor dtensor_from_local_ad_function(
    const paddle::Tensor& input,
    const phi::distributed::ProcessMesh& process_mesh,
    const phi::distributed::Placements& placements) {
#ifdef PADDLE_WITH_DISTRIBUTE
  VLOG(3) << "Running AD API: "
          << "dtensor_from_local dygraph";
  // Dygraph Record Event
  phi::RecordEvent dygraph_entrance_record_event(
      "dtensor_from_local dygraph", phi::TracerEventType::Communication, 1);

  // Get Input AutoGradMeta
  egr::AutogradMeta* input_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(input);
  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad =
      egr::EagerUtils::ComputeRequireGrad(trace_backward, input_autograd_meta);

  // Node Declaration
  std::shared_ptr<DtensorFromLocalGradNode> grad_node;

  // Set grad_node before API Call
  if (require_any_grad) {
    phi::RecordEvent node_creation_record_event(
        "dtensor_from_local node_creation",
        phi::TracerEventType::Communication,
        1);

    // Node Construction
    grad_node = std::shared_ptr<DtensorFromLocalGradNode>(
        new DtensorFromLocalGradNode(1, 1));  // NOLINT
  }

  auto dense_tensor_ptr =
      std::static_pointer_cast<phi::DenseTensor>(input.impl());

  auto global_dims = common::vectorize(dense_tensor_ptr->dims());
  for (size_t i = 0; i < placements.size(); i++) {
    auto placement = placements[i];
    if (placement->is_shard()) {
      auto shard_dim =
          dynamic_cast<const phi::distributed::Shard&>(*placement).get_dim();
      global_dims[shard_dim] = global_dims[shard_dim] * process_mesh.shape()[i];
    }
  }

  auto dist_out_ptr = std::make_shared<phi::distributed::DistTensor>(
      dense_tensor_ptr,
      common::make_ddim(global_dims),
      process_mesh,
      placements);

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
    grad_node->SetTensorWrapperNoNeedBuffer_Output(out);
  }

  return out;
#else
  PADDLE_THROW(common::errors::Unavailable(
      "DtensorFromLocal is not supported in this version of Paddle. Try to "
      "recompile it "
      "with WITH_DISTRIBUTE=ON and reinstall this package."));
  return paddle::Tensor();
#endif
}
