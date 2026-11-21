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

// paddle_comm.cc: paddle_nccl_comm, which gets the ncclComm_t that Paddle's
// ProcessGroup already holds for a ring id, through
//   ProcessGroupMapFromGid::getInstance()->get(ring_id)
//   -> ProcessGroupNCCL::GetOrCreateCommContext(place, CommType::ALLTOALL)
//   -> NCCLCommContext::GetNcclComm()
// If the group has no NCCL comm yet, GetOrCreateCommContext creates it.
// On failure it returns non-zero and *out_err holds the reason.
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
        "paddle::distributed::ProcessGroupMapFromGid::getInstance() returned "
        "null";
    return -1;
  }
  paddle::distributed::ProcessGroup* process_group =
      process_groups->get(ring_id);
  if (process_group == nullptr) {
    *out_err =
        "Paddle's ProcessGroupMapFromGid has no ring id " +
        std::to_string(ring_id) +
        " (the group is not registered on the C++ side yet? pass the .id of "
        "the Paddle group object)";
    return -1;
  }
  auto* nccl_process_group =
      dynamic_cast<paddle::distributed::ProcessGroupNCCL*>(process_group);
  if (nccl_process_group == nullptr) {
    *out_err = "ring id " + std::to_string(ring_id) +
               " is not a ProcessGroupNCCL (backend=" +
               process_group->GetBackendName() + ")";
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
               ": GetOrCreateCommContext(GPUPlace(" +
               std::to_string(device_id) + ")) threw: " + ex.what();
    return -1;
  }
  if (comm_context == nullptr) {
    *out_err = "ring id " + std::to_string(ring_id) +
               ": GetOrCreateCommContext returned null";
    return -1;
  }
  ncclComm_t comm = comm_context->GetNcclComm();
  if (comm == nullptr) {
    *out_err = "ring id " + std::to_string(ring_id) +
               ": NCCLCommContext::GetNcclComm() returned null";
    return -1;
  }
  *out_comm = comm;
  return 0;
}

}  // namespace gin_overlap
