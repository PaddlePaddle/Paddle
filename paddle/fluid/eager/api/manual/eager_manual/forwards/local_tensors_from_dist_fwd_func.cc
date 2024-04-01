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
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
#include "paddle/fluid/eager/api/manual/eager_manual/nodes/local_tensors_from_dist_node.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/placement_types.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"

std::vector<paddle::Tensor> local_tensors_from_dist_ad_function(
    const paddle::Tensor& dist_tensor,
    const std::vector<phi::distributed::ProcessMesh>& local_meshes,
    const phi::distributed::Placements local_placements,
    const phi::distributed::ProcessMesh& global_mesh,
    const phi::distributed::Placements& global_placements,
    int local_index) {
#ifdef PADDLE_WITH_DISTRIBUTE
  VLOG(3) << "Running AD API: "
          << "local_tensors_from_dist dygraph";
  // Dygraph Record Event
  paddle::platform::RecordEvent dygraph_entrance_record_event(
      "local_tensors_from_dist dygraph",
      paddle::platform::TracerEventType::Communication,
      1);

  // Get Input AutoGradMeta
  egr::AutogradMeta* input_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(dist_tensor);
  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad =
      egr::EagerUtils::ComputeRequireGrad(trace_backward, input_autograd_meta);

  // Node Declaration
  std::shared_ptr<LocalTensorsFromDistGradNode> grad_node;

  // Set grad_node before API Call
  if (require_any_grad) {
    paddle::platform::RecordEvent node_creation_record_event(
        "local_tensors_from_dist node_creation",
        paddle::platform::TracerEventType::Communication,
        1);

    // Node Construction
    grad_node = std::shared_ptr<LocalTensorsFromDistGradNode>(
        new LocalTensorsFromDistGradNode(1, 1));  // NOLINT

    // Set Attributes used in backward
    grad_node->SetAttribute_global_mesh(global_mesh);
    grad_node->SetAttribute_global_placements(global_placements);
    grad_node->SetAttribute_local_index(local_index);

    // Set TensorWrappers for Forward Inputs if needed
    grad_node->SetTensorWrapperNoNeedBuffer_Input(dist_tensor);
  }

  // Forward API Call
  std::vector<paddle::Tensor> api_result;
  common::DDim local_shape(dist_tensor.local_dims());
  for (int i = 0, n = local_placements.size(); i < n; ++i) {
    phi::distributed::Placement& placement = local_placements[i];
    if (placement.is_shard()) {
      int shard_dim = placement.get_dim();
      int local_dim_size = local_shape[shard_dim];
      local_shape[shard_dim] =
          local_dim_size * local_meshes[0].dim_size(shard_dim);
    }
  }
  for (int i = 0, n = local_meshes.size(); i < n; ++i) {
    phi::distributed::ProcessMesh& local_mesh = local_meshes[i];
    std::shared_ptr<phi::distributed::DistTensor> local_tensor_ptr =
        std::make_shared<phi::distributed::DistTensor>(
            dist_tensor.value() local_shape, local_mesh, local_placements);
    api_result.push_back(paddle::Tensor(local_tensor_ptr));
  }

  // reshard_func(input, api_result, dist_attr);
  // auto dist_out_ptr = paddle::reshard(input, dist_attr);
  // auto api_result = paddle::Tensor(dist_out_ptr);

  // Get Outputs
  auto& out = api_result;

  // Get Output AutoGradMeta
  egr::AutogradMeta* out_autograd_meta = egr::EagerUtils::autograd_meta(&out);

  // Set grad_node after API call
  if (require_any_grad) {
    egr::EagerUtils::PassStopGradient(false, out_autograd_meta);

    // SetGradOutMeta & SetEdges
    grad_node->SetGradOutMeta(dist_tensor, 0);
    // SetOutRank & SetHistory & SetGradInMeta
    if (out_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(out_autograd_meta, 0);
      egr::EagerUtils::SetHistory(out_autograd_meta, grad_node);
    }
    grad_node->SetGradInMeta(out, 0);
  }

  return out;
#else
  PADDLE_THROW(phi::errors::Unavailable(
      "Reshard is not supported in this version of Paddle. Try to recompile it "
      "with WITH_DISTRIBUTE=ON and reinstall this package."));
  return paddle::Tensor();
#endif
}
