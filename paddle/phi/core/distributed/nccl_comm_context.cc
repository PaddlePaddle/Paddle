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

#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {
namespace distributed {

NCCLCommContext::NCCLCommContext(int rank, int size, ncclUniqueId nccl_id)
    : CommContext(rank, size) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclCommInitRank(&nccl_comm_, size_, nccl_id, rank_));
}

void NCCLCommContext::Broadcast(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                int root,
                                gpuStream_t stream) {
  phi::dynload::ncclBroadcast(in_tensor.data(),
                              out_tensor->data(),
                              in_tensor.numel(),
                              ToNCCLDataType(in_tensor.type()),
                              root,
                              nccl_comm_,
                              stream);
}

}  // namespace distributed
}  // namespace phi
