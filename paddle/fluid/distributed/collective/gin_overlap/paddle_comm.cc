// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// paddle_comm.cc —— paddle_nccl_comm：按 ring id 取 Paddle ProcessGroup 已有的
// ncclComm_t，路径是
//   ProcessGroupMapFromGid::getInstance()->get(ring_id)
//   → ProcessGroupNCCL::GetOrCreateCommContext(place, CommType::ALLTOALL)
//   → NCCLCommContext::GetNcclComm()
// 这个组还没有 NCCL comm 时，GetOrCreateCommContext 会当场创建。
// 取不到时返回非 0，*out_err 是原因。
#include <exception>
#include <string>

#include <cuda_runtime.h>
#include <nccl.h>

#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"

namespace gin_overlap {

int paddle_nccl_comm(int ring_id,
                     int device_id,
                     ncclComm_t* out_comm,
                     int* out_size,
                     int* out_rank,
                     std::string* out_err) {
  *out_comm = nullptr;
  *out_size = 0;
  *out_rank = 0;
  auto process_groups =
      paddle::distributed::ProcessGroupMapFromGid::getInstance();
  if (process_groups == nullptr) {
    *out_err =
        "paddle::distributed::ProcessGroupMapFromGid::getInstance() 返回空";
    return -1;
  }
  paddle::distributed::ProcessGroup* process_group =
      process_groups->get(ring_id);
  if (process_group == nullptr) {
    *out_err =
        "Paddle 的 ProcessGroupMapFromGid 里没有 ring id " +
        std::to_string(ring_id) +
        "（这个通信组还没在 C++ 侧注册？确认传的是 paddle 通信组对象的 .id）";
    return -1;
  }
  auto* nccl_process_group =
      dynamic_cast<paddle::distributed::ProcessGroupNCCL*>(process_group);
  if (nccl_process_group == nullptr) {
    *out_err = "ring id " + std::to_string(ring_id) +
               " 对应的 ProcessGroup 不是 ProcessGroupNCCL（backend=" +
               process_group->GetBackendName() + "）";
    return -1;
  }
  *out_size = process_group->GetSize();
  *out_rank = process_group->GetRank();
  phi::distributed::NCCLCommContext* comm_context = nullptr;
  try {
    comm_context = nccl_process_group->GetOrCreateCommContext(
        phi::GPUPlace(device_id), phi::distributed::CommType::ALLTOALL);
  } catch (const std::exception& ex) {
    *out_err = "ring id " + std::to_string(ring_id) +
               " 的 GetOrCreateCommContext(GPUPlace(" +
               std::to_string(device_id) + ")) 抛异常：" + ex.what();
    return -1;
  }
  if (comm_context == nullptr) {
    *out_err = "ring id " + std::to_string(ring_id) +
               " 的 GetOrCreateCommContext 返回空";
    return -1;
  }
  ncclComm_t comm = comm_context->GetNcclComm();
  if (comm == nullptr) {
    *out_err = "ring id " + std::to_string(ring_id) +
               " 的 NCCLCommContext::GetNcclComm() 返回空";
    return -1;
  }
  *out_comm = comm;
  return 0;
}

}  // namespace gin_overlap
