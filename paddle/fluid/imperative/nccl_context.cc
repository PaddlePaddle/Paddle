//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/nccl_context.h"

namespace paddle {
namespace imperative {

void NCCLParallelContext::BcastNCCLID(ncclUniqueId *nccl_id, int root) {
  // TODO(Yancey1989): bcast the unique nccl ID by gRPC
  return;
}

void NCCLParallelContext::Init() {
  ncclUniqueId nccl_id;
  ncclComm_t comm;
  if (strategy_.local_rank_ == 0) {
    // generate the unique ncclid on the root worker
    platform::dynload::ncclGetUniqueId(&nccl_id);
    BcastNCCLID(&nccl_id, 0);
  } else {
    BcastNCCLID(&nccl_id, 0);
  }
  int gpu_id = boost::get<platform::CUDAPlace>(place_).device;
  PADDLE_ENFORCE(cudaSetDevice(gpu_id));
  PADDLE_ENFORCE(platform::dynload::ncclCommInitRank(
      &comm, strategy_.nranks_, nccl_id, strategy_.local_rank_));

  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto *dev_ctx = static_cast<platform::CUDADeviceContext *>(pool.Get(place_));
  dev_ctx->set_nccl_comm(comm);
}

}  //  namespace imperative
}  //  namespace paddle
