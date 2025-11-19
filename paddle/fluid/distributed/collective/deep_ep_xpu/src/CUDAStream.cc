// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/collective/deep_ep_xpu/include/CUDAStream.h"
#include "paddle/fluid/distributed/collective/deep_ep_xpu/kernels/exception.h"
// #include "paddle/fluid/distributed/collective/process_group_nccl.h"
#include "paddle/fluid/distributed/collective/process_group_bkcl.h"
#include "paddle/phi/common/place.h"

namespace deep_ep::detail {

cudaStream_t GetCalcStreamFromGroup(int context_ring_id) {
  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
  paddle::distributed::ProcessGroup* pg = map->get(context_ring_id);
  const auto& place = phi::XPUPlace(device_id);
  const auto& calc_ctx = reinterpret_cast<phi::XPUContext*>(
      reinterpret_cast<paddle::distributed::ProcessGroupBKCL*>(pg)
          ->GetDeviceContext(place, true));
  return reinterpret_cast<cudaStream_t>(calc_ctx->stream());
}

cudaStream_t GetCommStreamFromGroup(int context_ring_id) {
  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
  paddle::distributed::ProcessGroup* pg = map->get(context_ring_id);
  const auto& place = phi::XPUPlace(device_id);
  const auto& comm_ctx =
      reinterpret_cast<paddle::distributed::ProcessGroupBKCL*>(pg)
          ->GetOrCreateCommContext(place, phi::distributed::CommType::ALLTOALL);
  return reinterpret_cast<cudaStream_t>(comm_ctx->GetStream());
}

}  // namespace deep_ep::detail
