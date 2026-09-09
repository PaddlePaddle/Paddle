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

#include <unordered_map>

#include "paddle/common/macros.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/core/distributed/comm_context.h"
#include "paddle/phi/core/distributed/nccl_config.h"

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
  NCCLCommContext(
      int rank,
      int size,
      ncclUniqueId nccl_id,
      int nccl_comm_init_option = 0,
      std::shared_ptr<phi::distributed::NCCLConfig> nccl_config_ptr = nullptr);
  ~NCCLCommContext() override = default;

  int GetNcclVersion();

  ncclComm_t GetNcclComm();

  void CreateNCCLComm(
      ncclUniqueId nccl_id,
      std::shared_ptr<phi::distributed::NCCLConfig> nccl_config_ptr = nullptr);

  void DestroyNCCLComm();

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

  void Broadcast(DenseTensor* out_tensor,
                 const DenseTensor& in_tensor,
                 int root,
                 gpuStream_t stream);

  void Send(const DenseTensor& in_tensor,
            const int64_t& count,
            const int& peer,
            gpuStream_t stream);

  void Recv(DenseTensor* out_tensor,
            const int64_t& count,
            const int& peer,
            gpuStream_t stream);

  void ReduceScatter(DenseTensor* out_tensor,
                     const DenseTensor& in_tensor,
                     ncclRedOp_t reduce_type,
                     gpuStream_t stream);

  void AllGather(DenseTensor* out_tensor,
                 const DenseTensor& in_tensor,
                 gpuStream_t stream);

  void AllReduce(DenseTensor* out_tensor,
                 const DenseTensor& in_tensor,
                 ncclRedOp_t reduce_type,
                 gpuStream_t stream);

  void Reduce(DenseTensor* out_tensor,
              const DenseTensor& in_tensor,
              ncclRedOp_t reduce_type,
              int root,
              gpuStream_t stream);

  void GroupStart();

  void GroupEnd();

  // Registers a device buffer as a NCCL symmetric memory window, which is what
  // makes the zero-SM paths (NCCL_CTA_POLICY_ZERO) usable, and returns the
  // window handle as an opaque pointer. Registration is collective: every rank
  // must register buffers of the same size in the same order, and a single
  // collective call must have either all of its buffers registered or none.
  // `ptr` and `size` must be aligned to kNCCLWindowAlignment. Repeated calls
  // for the same `ptr` return the cached handle; returns nullptr when the
  // loaded NCCL provides no window API.
  void* RegisterWindow(void* ptr, size_t size, int win_flags);

  // Deregisters a previously registered buffer. No-op for unknown pointers.
  void DeregisterWindow(void* ptr);

  // Deregisters every window owned by this communicator, before it is
  // destroyed.
  void DeregisterAllWindows();

  // True when [ptr, ptr + size) lies inside a window registered here, i.e. when
  // a collective over that buffer may take a symmetric-memory path.
  bool IsRegistered(const void* ptr, size_t size) const;

  // True when the loaded NCCL provides the single-call all-to-all, the only
  // all-to-all entry point that can reach the zero-SM path. A group of
  // Send/Recv calls always runs a point-to-point device kernel instead.
  bool IsAllToAllAvailable() const;

  // Symmetric all-to-all: rank j receives in_tensor[i * count, (i + 1) * count)
  // from rank i, so every rank must contribute and receive the same count.
  // Only call it when IsAllToAllAvailable() is true.
  void AllToAll(DenseTensor* out_tensor,
                const DenseTensor& in_tensor,
                gpuStream_t stream);

  static constexpr size_t kNCCLWindowAlignment = 4096;

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

  int nranks;
  int myrank;
  int param;

  // A registered window: its NCCL handle and the byte range it covers, the
  // range being what lets IsRegistered() accept a slice of a registered buffer.
  struct RegisteredWindow {
    void* handle{nullptr};
    size_t size{0};
  };

  // Keyed by buffer pointer, so that registration is idempotent and every
  // window can be released before the communicator is destroyed.
  std::unordered_map<void*, RegisteredWindow> windows_;
};

}  // namespace distributed
}  // namespace phi
