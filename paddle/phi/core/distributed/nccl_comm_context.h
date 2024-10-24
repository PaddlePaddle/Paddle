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

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif

#include "paddle/common/macros.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/core/distributed/comm_context.h"

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
  NCCLCommContext(int rank,
                  int size,
                  ncclUniqueId nccl_id,
                  int nccl_comm_init_option = 0);
  ~NCCLCommContext() override = default;

  int GetNcclVersion();

  ncclComm_t GetNcclComm();

  gpuStream_t GetStream();

  gpuEvent_t GetComputeEvent();

  void SetComputeEvent(
      std::shared_ptr<std::remove_pointer<phi::gpuEvent_t>::type>&&
          compute_event);

  gpuEvent_t GetCommEvent();

  void SetCommEvent(
      std::shared_ptr<std::remove_pointer<phi::gpuEvent_t>::type>&& comm_event);

  phi::GPUContext* GetDevContext();

  void SetDevContext(std::unique_ptr<phi::GPUContext>&& dev_ctx);

  void Broadcast(phi::DenseTensor* out_tensor,
                 const phi::DenseTensor& in_tensor,
                 int root,
                 gpuStream_t stream);

  void Send(const phi::DenseTensor& in_tensor,
            const int64_t& count,
            const int& peer,
            gpuStream_t stream);

  void Recv(phi::DenseTensor* out_tensor,
            const int64_t& count,
            const int& peer,
            gpuStream_t stream);

  void ReduceScatter(phi::DenseTensor* out_tensor,
                     const phi::DenseTensor& in_tensor,
                     ncclRedOp_t reduce_type,
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

  void GroupStart();

  void GroupEnd();

#if NCCL_VERSION_CODE >= 21100
  // Creates a new reduction operator which pre-multiplies input values by a
  // given scalar locally before reducing them with peer values via summation.
  void RedOpCreatePreMulSum(ncclRedOp_t* op,
                            void* scalar,
                            ncclDataType_t dtype,
                            ncclScalarResidence_t residence);

  // Destroys the reduction operator op. The operator must have been created by
  // ncclRedOpCreatePreMul with the matching communicator comm.
  void RedOpDestroy(ncclRedOp_t op);
#endif

 private:
  DISABLE_COPY_AND_ASSIGN(NCCLCommContext);

  int nccl_version_;

  ncclComm_t nccl_comm_;

  std::unique_ptr<phi::GPUContext> dev_ctx_;

  // used for comm wait compute, compute_stream-->event-->comm_stream
  std::shared_ptr<std::remove_pointer<phi::gpuEvent_t>::type> compute_event_;

  // used for compute wait comm, comm_stream-->event-->compute_stream
  std::shared_ptr<std::remove_pointer<phi::gpuEvent_t>::type> comm_event_;
};

}  // namespace distributed
}  // namespace phi
