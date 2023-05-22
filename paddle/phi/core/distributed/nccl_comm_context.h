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
#pragma once

#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/core/distributed/comm_context.h"
#include "paddle/phi/core/macros.h"

#if defined(PADDLE_WITH_RCCL)
#include "paddle/phi/backends/dynload/rccl.h"
#else
#include "paddle/phi/backends/dynload/nccl.h"
#endif

namespace phi {
class DenseTensor;
namespace distributed {

class NCCLCommContext final : public CommContext {
 public:
  NCCLCommContext(int rank, int size, ncclUniqueId nccl_id);

  ncclComm_t GetNcclComm();

  void Broadcast(phi::DenseTensor* out_tensor,
                 const phi::DenseTensor& in_tensor,
                 int root,
                 gpuStream_t stream);
  void Send(const phi::DenseTensor& in_tensor,
            const int& peer,
            gpuStream_t stream);

  void Recv(phi::DenseTensor* out_tensor, const int& peer, gpuStream_t stream);

  void ReduceScatter(phi::DenseTensor* out_tensor,
                     const phi::DenseTensor& in_tensor,
                     gpuStream_t stream);

  void AllGather(phi::DenseTensor* out_tensor,
                 const phi::DenseTensor& in_tensor,
                 gpuStream_t stream);

  void AllReduce(phi::DenseTensor* out_tensor,
                 const phi::DenseTensor& in_tensor,
                 ncclRedOp_t reduce_type,
                 gpuStream_t stream);

  void Reduce(phi::DenseTensor* out_tensor,
              const phi::DenseTensor& in_tensor,
              ncclRedOp_t reduce_type,
              int root,
              gpuStream_t stream);

 private:
  DISABLE_COPY_AND_ASSIGN(NCCLCommContext);

  ncclComm_t nccl_comm_;
};

}  // namespace distributed
}  // namespace phi
