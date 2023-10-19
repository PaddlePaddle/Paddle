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

#include "glog/logging.h"
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
#include "paddle/fluid/eager/api/manual/eager_manual/nodes/nodes.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_function.h"
#include "paddle/phi/core/enforce.h"

paddle::small_vector<std::vector<paddle::Tensor>,
                     egr::kSlotSmallVectorSize>  // NOLINT
ReshardGradNode::operator()(
    paddle::small_vector<std::vector<paddle::Tensor>,
                         egr::kSlotSmallVectorSize>& grads,
    bool create_graph,
    bool is_new_grad) {
  VLOG(3) << "Running API GRAD: "
          << "reshard_grad";

  // Apply Gradient Hooks
  auto hooked_grad = ApplyGradientHooks(grads);

  // Collect GradIn Tensors, Attrs and Recovered TensorWrappers
  auto input = egr::EagerUtils::RecoverTensorWrapper(&this->input_);
  const auto& dist_attr =
      std::static_pointer_cast<phi::distributed::DistTensor>(input.impl())
          ->dist_attr();
  auto& grad_out = hooked_grad[0][0];
  // Prepare Grad function call

  const auto& out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
      returns(1);
  for (int i = 0; i < 1; ++i) {
    out_metas[i].size() == 0 ? returns[i].resize(1)
                             : returns[i].resize(out_metas[i].size());
  }

  auto& grad_input = returns[0][0];

  VLOG(5) << "Running C++ API: "
          << "reshard_func";

  // Backward call reshard_func function
  // reshard_func(grad_out, grad_input, dist_attr);
  auto dev_ctx = phi::DeviceContextPool::Instance().Get(grad_out.place());
  std::shared_ptr<phi::distributed::DistTensor> dist_out_ptr = nullptr;
  if (phi::distributed::DistTensor::classof(grad_out.impl().get())) {
    auto tensor_grad_out = grad_out.impl();
    if (tensor_grad_out) {
      phi::distributed::DistTensor* dist_tensor =
          static_cast<phi::distributed::DistTensor*>(tensor_grad_out.get());
      if (dist_tensor->dist_attr() != dist_attr) {
        VLOG(6) << "reshard func, reshard tensor from "
                << dist_tensor->dist_attr() << " to " << dist_attr;
        auto* func = phi::distributed::ChooseProperReshardFunction(*dist_tensor,
                                                                   dist_attr);
        dist_out_ptr = func->Eval(dev_ctx, *dist_tensor, dist_attr);
      } else {
        dist_out_ptr = std::static_pointer_cast<phi::distributed::DistTensor>(
            tensor_grad_out);
      }
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "The input tensor of shard function should be "
        "``phi::distributed::DistTensor``. "
        "However it's %s",
        typeid(grad_out.impl().get()).name()));
  }
  grad_input.set_impl(dist_out_ptr);

  VLOG(5) << "Finish C++ API: reshard_func";
  VLOG(6) << "gradnode_ptr = " << this;

  return returns;
}
