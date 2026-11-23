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

#include <memory>
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

// Properties of the device communicator (ncclDevComm) of a NCCLCommContext,
// returned by NCCLCommContext::GetOrCreateDevComm().
struct NCCLDevCommInfo {
  // Points to the ncclDevComm (nccl_device.h). Valid until the communicator is
  // destroyed or recreated.
  const void* dev_comm{nullptr};
  // Rank and size of the communicator.
  int rank{0};
  int nranks{0};
  // Index of this rank in its LSA team (the ranks that can load and store each
  // other's windows directly) and the size of that team. An LSA team is a run
  // of consecutive ranks: rank r belongs to team r / lsa_size.
  int lsa_rank{0};
  int lsa_size{0};
  // Number of GIN contexts the device communicator was created with.
  int gin_context_count{0};
};

// A window over pinned host memory, returned by
// NCCLCommContext::AllocHostWindow().
struct NCCLHostWindow {
  // Window handle, usable wherever a handle from RegisterWindow() is.
  void* window{nullptr};
  // Start of the memory. The CPU and the GPU of the communicator both access
  // it through this address.
  void* ptr{nullptr};
  // Size of the memory in bytes.
  size_t bytes{0};
};

class NCCLCommContext final : public CommContext {
 public:
  NCCLCommContext(
      int rank,
      int size,
      ncclUniqueId nccl_id,
      int nccl_comm_init_option = 0,
      std::shared_ptr<phi::distributed::NCCLConfig> nccl_config_ptr = nullptr);
  ~NCCLCommContext() override;

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

  // Returns the handle of the window registered at `ptr` by RegisterWindow(),
  // or nullptr when `ptr` is not the start of a registered buffer.
  void* GetWindow(const void* ptr) const;

  // Looks up a window handle returned by RegisterWindow(). Returns false when
  // the handle is not registered here; otherwise sets *ptr and *size to the
  // registered buffer.
  bool GetWindowRange(void* window, void** ptr, size_t* size) const;

  // True when Paddle was built with PADDLE_WITH_GIN_OVERLAP (the NCCL device
  // API: NCCL 2.30.7 or newer with nccl_device.h, CUDA 13.0 or newer, sm_90 or
  // newer) and the loaded NCCL provides the host entry points of that API.
  static bool IsDevCommSupported();

  // Returns the device communicator (ncclDevComm) of this communicator and
  // creates it on the first call. Creation is collective: every rank must make
  // the first call, in the same order relative to other collective calls. The
  // device communicator has full GIN connectivity, one GIN signal and no GIN
  // counter; gin_context_count is a hint for the number of GIN contexts, the
  // number created is NCCLDevCommInfo::gin_context_count. Later calls return
  // the cached device communicator and must pass the same hint. Windows can be
  // registered before or after the device communicator is created. It is
  // destroyed by DestroyNCCLComm() before the communicator.
  NCCLDevCommInfo GetOrCreateDevComm(int gin_context_count);

  // True once GetOrCreateDevComm() has created the device communicator.
  bool HasDevComm() const;

  // The device communicator created by GetOrCreateDevComm(). Raises when it
  // has not been created.
  const NCCLDevCommInfo& GetDevCommInfo() const;

  // Device address of byte `offset` of `window` in the memory of the rank with
  // index `lsa_peer` in this rank's LSA team. This GPU can load from and store
  // to that address. `window` is a handle returned by RegisterWindow() or
  // AllocHostWindow(); `offset` may equal the window size. Requires the device
  // communicator (GetOrCreateDevComm()).
  void* LsaPointer(void* window, int lsa_peer, size_t offset);

  // Allocates pinned host memory on the host NUMA node of this communicator's
  // GPU, mapped for the GPU and the CPU at the same address, zeroes it, and
  // registers it as a window with NCCL_WIN_COLL_SYMMETRIC |
  // NCCL_WIN_STRICT_ORDERING. Collective, like RegisterWindow(): every rank
  // must call it with the same `bytes`. The size is `bytes` rounded up to the
  // allocation granularity. The window is in the same table as the windows of
  // RegisterWindow(). FreeHostWindow() releases it; DestroyNCCLComm() and a
  // recreation of the communicator release every host window.
  NCCLHostWindow AllocHostWindow(size_t bytes);

  // Deregisters a window returned by AllocHostWindow() and frees its memory.
  // Collective.
  void FreeHostWindow(void* window);

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

  // Destroys the device communicator while nccl_comm_ is still alive.
  void DestroyDevComm();

  // Unmaps and frees the memory of every host window. Their windows must have
  // been deregistered already.
  void FreeAllHostWindowMemory();

  // State of the NCCL device API and of window lookups by handle: the device
  // communicator, the window handle -> buffer map, the LSA addresses of
  // windows and the host window allocations. Defined in nccl_comm_context.cc
  // so that this header does not include nccl_device.h.
  struct DeviceApiState;
  std::unique_ptr<DeviceApiState> device_api_;
};

}  // namespace distributed
}  // namespace phi
