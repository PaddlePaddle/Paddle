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

// PADDLE_WITH_GIN_OVERLAP (cmake/configure.cmake) means Paddle is built against
// the NCCL device API: NCCL 2.30.7 or newer with nccl_device.h, CUDA 13.0 or
// newer, an arch of sm_90 or newer.
#ifdef PADDLE_WITH_GIN_OVERLAP
#include <nccl_device.h>
#endif

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include "glog/logging.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/check/nccl_dynamic_check.h"
#include "paddle/phi/core/distributed/check/static_check.h"
#include "paddle/phi/core/distributed/nccl_tools.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/utils/data_type.h"

#ifdef PADDLE_WITH_GIN_OVERLAP
#include "paddle/phi/backends/dynload/cuda_driver.h"
#endif

namespace phi::distributed {

// set this flag to `true` and recompile to enable dynamic checks
constexpr bool FLAGS_enable_nccl_dynamic_check = false;

struct NCCLCommContext::DeviceApiState {
  // Window handle -> start of the registered buffer (the key of windows_).
  std::unordered_map<void*, void*> window_buffers;
  // Window handle -> device address of offset 0 of the window on each LSA
  // index, filled on first use (nullptr = not looked up yet).
  std::unordered_map<void*, std::vector<void*>> lsa_window_bases;
  bool has_dev_comm{false};
  int requested_gin_context_count{0};
  NCCLDevCommInfo dev_comm_info;
#ifdef PADDLE_WITH_GIN_OVERLAP
  ncclDevComm dev_comm{};
  // Memory of a host window.
  struct HostMemory {
    CUmemGenericAllocationHandle handle{0};
    void* ptr{nullptr};
    size_t bytes{0};
  };
  // Window handle -> memory, for the windows of AllocHostWindow().
  std::unordered_map<void*, HostMemory> host_windows;
#endif
};

#ifdef PADDLE_WITH_GIN_OVERLAP
namespace {

// Unmaps and frees the memory of a host window.
void ReleaseHostMemory(CUmemGenericAllocationHandle handle,
                       void* ptr,
                       size_t bytes) {
  auto address = reinterpret_cast<CUdeviceptr>(ptr);
  if (address != 0) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemUnmap(address, bytes));
  }
  if (handle != 0) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemRelease(handle));
  }
  if (address != 0) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemAddressFree(address, bytes));
  }
}

}  // namespace
#endif

NCCLCommContext::NCCLCommContext(
    int rank,
    int size,
    ncclUniqueId nccl_id,
    int nccl_comm_init_option,
    std::shared_ptr<phi::distributed::NCCLConfig> nccl_config_ptr)
    : CommContext(rank, size),
      nccl_version_(0),
      nccl_comm_(nullptr),
      nranks(size_),
      myrank(rank_),
      param(nccl_comm_init_option),
      device_api_(std::make_unique<DeviceApiState>()) {
  this->CreateNCCLComm(nccl_id, nccl_config_ptr);
  NCCL_CHECK(phi::dynload::ncclGetVersion(&nccl_version_));
}

NCCLCommContext::~NCCLCommContext() = default;

void NCCLCommContext::CreateNCCLComm(
    ncclUniqueId nccl_id,
    std::shared_ptr<phi::distributed::NCCLConfig> nccl_config_ptr) {
  // A restart replaces the communicator on this very context, and a window
  // handle only means anything to the communicator it was registered against.
  // Drop the bookkeeping here so that RegisterWindow() cannot hand a stale
  // handle back from its cache: deregister while the old communicator is still
  // alive, or simply forget the windows when it is already gone.
  if (nccl_comm_ != nullptr) {
    DestroyDevComm();
    DeregisterAllWindows();
    FreeAllHostWindowMemory();
  } else {
    windows_.clear();
    device_api_->window_buffers.clear();
    device_api_->lsa_window_bases.clear();
    FreeAllHostWindowMemory();
    device_api_->has_dev_comm = false;
    device_api_->requested_gin_context_count = 0;
    device_api_->dev_comm_info = NCCLDevCommInfo{};
  }
  if (param > 0 && phi::dynload::ncclCommInitRank2.IsValid()) {
    LOG(WARNING) << "Creating modified qp with ncclCommInitRank2.";
    NCCL_CHECK(phi::dynload::ncclCommInitRank2(
        &nccl_comm_, nranks, nccl_id, myrank, param));
  } else {
    if (param > 0) {
      LOG(WARNING) << "ncclCommInitRank2 is not supported.";
    }
    if (nccl_config_ptr != nullptr &&
        phi::dynload::ncclCommInitRankConfigMemOpt.IsValid()) {
      NCCL_CHECK(phi::dynload::ncclCommInitRankConfigMemOpt(
          &nccl_comm_,
          nranks,
          nccl_id,
          myrank,
          nccl_config_ptr->GetOrigin(),
          nccl_config_ptr->GetMemOpt()));
    } else {
      if (nccl_config_ptr != nullptr) {
        LOG(WARNING) << "ncclCommInitRankConfigMemOpt is not supported.";
      }
#if defined(PADDLE_WITH_NCCL) && NCCL_VERSION_CODE >= 23007
      // Without the internal MemOpt entry point, a requested ncclConfig_t (for
      // instance a CTAPolicy) would be dropped silently, so fall back to the
      // upstream config-aware initializer whenever there is a config to pass.
      if (nccl_config_ptr != nullptr &&
          nccl_config_ptr->GetOrigin() != nullptr &&
          phi::dynload::ncclCommInitRankConfig.IsValid()) {
        NCCL_CHECK(
            phi::dynload::ncclCommInitRankConfig(&nccl_comm_,
                                                 nranks,
                                                 nccl_id,
                                                 myrank,
                                                 nccl_config_ptr->GetOrigin()));
        return;
      }
#endif
      NCCL_CHECK(
          phi::dynload::ncclCommInitRank(&nccl_comm_, nranks, nccl_id, myrank));
    }
  }
}

void NCCLCommContext::DestroyNCCLComm() {
  if (nccl_comm_ != nullptr) {
    DestroyDevComm();
    DeregisterAllWindows();
    FreeAllHostWindowMemory();
    NCCL_CHECK(phi::dynload::ncclCommDestroy(nccl_comm_));
    nccl_comm_ = nullptr;
  }
}

void* NCCLCommContext::RegisterWindow(void* ptr, size_t size, int win_flags) {
#if defined(PADDLE_WITH_NCCL) && NCCL_VERSION_CODE >= 23007
  if (!phi::dynload::ncclCommWindowRegister.IsValid()) {
    LOG(WARNING) << "ncclCommWindowRegister is not provided by the loaded NCCL "
                    "library, which the zero-SM paths need (NCCL 2.30.7 or "
                    "newer); buffer registration is skipped.";
    return nullptr;
  }
  auto iter = windows_.find(ptr);
  if (iter != windows_.end()) {
    return iter->second.handle;
  }
  PADDLE_ENFORCE_EQ(
      reinterpret_cast<uintptr_t>(ptr) % kNCCLWindowAlignment,
      0,
      common::errors::InvalidArgument(
          "The buffer registered as a NCCL symmetric memory window must be "
          "aligned to %d bytes, but got pointer %p.",
          kNCCLWindowAlignment,
          ptr));
  PADDLE_ENFORCE_EQ(
      size % kNCCLWindowAlignment,
      0,
      common::errors::InvalidArgument(
          "The size of a buffer registered as a NCCL symmetric memory window "
          "must be a multiple of %d bytes, but got %d.",
          kNCCLWindowAlignment,
          size));
  ncclWindow_t win = nullptr;
  NCCL_CHECK(phi::dynload::ncclCommWindowRegister(
      nccl_comm_, ptr, size, &win, win_flags));
  windows_.emplace(ptr, RegisteredWindow{reinterpret_cast<void*>(win), size});
  // Without symmetric memory support NCCL returns no window; GetWindowRange()
  // must then reject a null handle.
  if (win != nullptr) {
    device_api_->window_buffers[reinterpret_cast<void*>(win)] = ptr;
  }
  VLOG(3) << "Registered NCCL window for buffer " << ptr << " of size " << size
          << " with flags " << win_flags;
  return reinterpret_cast<void*>(win);
#else
  LOG(WARNING) << "Paddle was compiled against NCCL " << NCCL_VERSION_CODE
               << ", which has no symmetric memory window API. Buffer "
                  "registration is skipped.";
  return nullptr;
#endif
}

void NCCLCommContext::DeregisterWindow(void* ptr) {
#if defined(PADDLE_WITH_NCCL) && NCCL_VERSION_CODE >= 23007
  auto iter = windows_.find(ptr);
  if (iter == windows_.end()) {
    return;
  }
  if (phi::dynload::ncclCommWindowDeregister.IsValid()) {
    NCCL_CHECK(phi::dynload::ncclCommWindowDeregister(
        nccl_comm_, reinterpret_cast<ncclWindow_t>(iter->second.handle)));
  }
  device_api_->window_buffers.erase(iter->second.handle);
  device_api_->lsa_window_bases.erase(iter->second.handle);
  windows_.erase(iter);
#endif
}

void NCCLCommContext::DeregisterAllWindows() {
#if defined(PADDLE_WITH_NCCL) && NCCL_VERSION_CODE >= 23007
  if (phi::dynload::ncclCommWindowDeregister.IsValid()) {
    for (auto& item : windows_) {
      NCCL_CHECK(phi::dynload::ncclCommWindowDeregister(
          nccl_comm_, reinterpret_cast<ncclWindow_t>(item.second.handle)));
    }
  }
  windows_.clear();
  device_api_->window_buffers.clear();
  device_api_->lsa_window_bases.clear();
#endif
}

bool NCCLCommContext::IsRegistered(const void* ptr, size_t size) const {
  const auto* begin = static_cast<const char*>(ptr);
  for (const auto& item : windows_) {
    const auto* window_begin = static_cast<const char*>(item.first);
    if (begin >= window_begin &&
        begin + size <= window_begin + item.second.size) {
      return true;
    }
  }
  return false;
}

bool NCCLCommContext::GetWindowRange(void* window,
                                     void** base,
                                     size_t* bytes) const {
  auto iter = device_api_->window_buffers.find(window);
  if (iter == device_api_->window_buffers.end()) {
    return false;
  }
  *base = iter->second;
  *bytes = windows_.at(iter->second).size;
  return true;
}

bool NCCLCommContext::IsDevCommSupported() {
#ifdef PADDLE_WITH_GIN_OVERLAP
  // NCCL 2.28 / 2.29 already export the device API entry points, but their
  // ncclDevCommCreate rejects the requirements of newer headers, so the loaded
  // library is also checked for the version Paddle requires.
  static const int runtime_version = [] {
    int version = 0;
    return phi::dynload::ncclGetVersion(&version) == ncclSuccess ? version : 0;
  }();
  return runtime_version >= 23007 &&
         phi::dynload::ncclDevCommCreate.IsValid() &&
         phi::dynload::ncclDevCommDestroy.IsValid() &&
         phi::dynload::ncclGetLsaDevicePointer.IsValid() &&
         phi::dynload::ncclCommWindowRegister.IsValid();
#else
  return false;
#endif
}

NCCLDevCommInfo NCCLCommContext::GetOrCreateDevComm(int gin_context_count) {
#ifdef PADDLE_WITH_GIN_OVERLAP
  DeviceApiState& device_api = *device_api_;
  if (device_api.has_dev_comm) {
    PADDLE_ENFORCE_EQ(
        gin_context_count,
        device_api.requested_gin_context_count,
        common::errors::InvalidArgument(
            "The device communicator was created with a GIN context count "
            "hint of %d, but got %d.",
            device_api.requested_gin_context_count,
            gin_context_count));
    return device_api.dev_comm_info;
  }
  PADDLE_ENFORCE_EQ(
      IsDevCommSupported(),
      true,
      common::errors::Unavailable(
          "The loaded NCCL library is older than 2.30.7 or does not provide "
          "the device API entry points (ncclDevCommCreate, "
          "ncclDevCommDestroy, ncclGetLsaDevicePointer, "
          "ncclCommWindowRegister); NCCL 2.30.7 or newer is required."));
  PADDLE_ENFORCE_NOT_NULL(
      nccl_comm_,
      common::errors::PreconditionNotMet(
          "The NCCL communicator has not been created or has been destroyed."));
  PADDLE_ENFORCE_GT(gin_context_count,
                    0,
                    common::errors::InvalidArgument(
                        "The GIN context count hint must be positive, but "
                        "got %d.",
                        gin_context_count));
  ncclDevCommRequirements_t requirements =
      NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  requirements.ginContextCount = gin_context_count;
  requirements.ginSignalCount = 1;
  requirements.ginCounterCount = 0;
  NCCL_CHECK(phi::dynload::ncclDevCommCreate(
      nccl_comm_, &requirements, &device_api.dev_comm));
  device_api.has_dev_comm = true;
  device_api.requested_gin_context_count = gin_context_count;
  device_api.dev_comm_info.dev_comm = &device_api.dev_comm;
  device_api.dev_comm_info.rank = device_api.dev_comm.rank;
  device_api.dev_comm_info.nranks = device_api.dev_comm.nRanks;
  device_api.dev_comm_info.lsa_rank = device_api.dev_comm.lsaRank;
  device_api.dev_comm_info.lsa_size = device_api.dev_comm.lsaSize;
  device_api.dev_comm_info.gin_context_count =
      static_cast<int>(device_api.dev_comm.ginContextCount);
  VLOG(3) << "Created NCCL device communicator: rank "
          << device_api.dev_comm_info.rank << "/"
          << device_api.dev_comm_info.nranks << ", lsa rank "
          << device_api.dev_comm_info.lsa_rank << "/"
          << device_api.dev_comm_info.lsa_size << ", "
          << device_api.dev_comm_info.gin_context_count << " GIN contexts";
  return device_api.dev_comm_info;
#else
  PADDLE_THROW(common::errors::Unavailable(
      "Paddle was not built with PADDLE_WITH_GIN_OVERLAP (the NCCL device "
      "API: NCCL 2.30.7 or newer with nccl_device.h, CUDA 13.0 or newer, "
      "sm_90 or newer)."));
#endif
}

const NCCLDevCommInfo& NCCLCommContext::GetDevCommInfo() const {
  PADDLE_ENFORCE_EQ(device_api_->has_dev_comm,
                    true,
                    common::errors::PreconditionNotMet(
                        "The NCCL device communicator has not been created; "
                        "call GetOrCreateDevComm() on every rank first."));
  return device_api_->dev_comm_info;
}

void NCCLCommContext::DestroyDevComm() {
#ifdef PADDLE_WITH_GIN_OVERLAP
  DeviceApiState& device_api = *device_api_;
  if (device_api.has_dev_comm) {
    NCCL_CHECK(
        phi::dynload::ncclDevCommDestroy(nccl_comm_, &device_api.dev_comm));
  }
  device_api.dev_comm = ncclDevComm{};
#endif
  device_api_->has_dev_comm = false;
  device_api_->requested_gin_context_count = 0;
  device_api_->dev_comm_info = NCCLDevCommInfo{};
}

void* NCCLCommContext::LsaPointer(void* window, int lsa_index, size_t offset) {
#ifdef PADDLE_WITH_GIN_OVERLAP
  void* window_base = nullptr;
  size_t window_bytes = 0;
  PADDLE_ENFORCE_EQ(
      GetWindowRange(window, &window_base, &window_bytes),
      true,
      common::errors::InvalidArgument(
          "Window %p is not registered on this communicator.", window));
  PADDLE_ENFORCE_LE(offset,
                    window_bytes,
                    common::errors::InvalidArgument(
                        "Offset %d is past the end of window %p (%d bytes).",
                        offset,
                        window,
                        window_bytes));
  const int lsa_size = GetDevCommInfo().lsa_size;
  PADDLE_ENFORCE_EQ(
      lsa_index >= 0 && lsa_index < lsa_size,
      true,
      common::errors::InvalidArgument(
          "LSA index %d is out of range [0, %d).", lsa_index, lsa_size));
  std::vector<void*>& peer_bases = device_api_->lsa_window_bases[window];
  if (peer_bases.size() <= static_cast<size_t>(lsa_index)) {
    peer_bases.resize(static_cast<size_t>(lsa_index) + 1, nullptr);
  }
  if (peer_bases[lsa_index] == nullptr) {
    NCCL_CHECK(phi::dynload::ncclGetLsaDevicePointer(
        reinterpret_cast<ncclWindow_t>(window),
        0,
        lsa_index,
        &peer_bases[lsa_index]));
  }
  return static_cast<char*>(peer_bases[lsa_index]) + offset;
#else
  PADDLE_THROW(common::errors::Unavailable(
      "Paddle was not built with PADDLE_WITH_GIN_OVERLAP (the NCCL device "
      "API: NCCL 2.30.7 or newer with nccl_device.h, CUDA 13.0 or newer, "
      "sm_90 or newer)."));
#endif
}

NCCLHostWindow NCCLCommContext::AllocHostWindow(size_t bytes) {
#ifdef PADDLE_WITH_GIN_OVERLAP
  PADDLE_ENFORCE_EQ(phi::dynload::ncclCommWindowRegister.IsValid(),
                    true,
                    common::errors::Unavailable(
                        "ncclCommWindowRegister is not provided by the loaded "
                        "NCCL library (NCCL 2.30.7 or newer is required)."));
  PADDLE_ENFORCE_NOT_NULL(
      nccl_comm_,
      common::errors::PreconditionNotMet(
          "The NCCL communicator has not been created or has been destroyed."));
  PADDLE_ENFORCE_GT(bytes,
                    static_cast<size_t>(0),
                    common::errors::InvalidArgument(
                        "The host window size must be positive."));
  int device_id = 0;
  NCCL_CHECK(phi::dynload::ncclCommCuDevice(nccl_comm_, &device_id));
  CUdevice cu_device;
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuDeviceGet(&cu_device, device_id));
  int numa_id = -1;
  if (phi::dynload::cuDeviceGetAttribute(&numa_id,
                                         CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
                                         cu_device) != CUDA_SUCCESS ||
      numa_id < 0) {
    numa_id = 0;
  }
  CUmemAllocationProp alloc_prop;
  std::memset(&alloc_prop, 0, sizeof(alloc_prop));
  alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  alloc_prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  alloc_prop.location.id = numa_id;
  // The ranks of the same node map this memory through a file descriptor.
  alloc_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  size_t granularity = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemGetAllocationGranularity(
      &granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  constexpr size_t kMaxSize = std::numeric_limits<size_t>::max();
  size_t alloc_bytes = std::max(bytes, kNCCLWindowAlignment);
  PADDLE_ENFORCE_LE(alloc_bytes,
                    kMaxSize - (granularity - 1),
                    common::errors::InvalidArgument(
                        "The host window size %d overflows size_t once "
                        "rounded up to the allocation granularity %d.",
                        bytes,
                        granularity));
  alloc_bytes = (alloc_bytes + granularity - 1) / granularity * granularity;

  DeviceApiState::HostMemory memory;
  memory.bytes = alloc_bytes;
  bool is_mapped = false;
  void* window = nullptr;
  try {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cuMemCreate(&memory.handle, alloc_bytes, &alloc_prop, 0));
    CUdeviceptr address = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemAddressReserve(
        &address, alloc_bytes, granularity, 0, 0));
    memory.ptr = reinterpret_cast<void*>(address);
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cuMemMap(address, alloc_bytes, 0, memory.handle, 0));
    is_mapped = true;
    CUmemAccessDesc access_desc;
    std::memset(&access_desc, 0, sizeof(access_desc));
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device_id;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cuMemSetAccess(address, alloc_bytes, &access_desc, 1));
    access_desc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    access_desc.location.id = numa_id;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cuMemSetAccess(address, alloc_bytes, &access_desc, 1));
    // The memory must be cleared before it is registered.
    std::memset(memory.ptr, 0, alloc_bytes);
    window = RegisterWindow(memory.ptr,
                            alloc_bytes,
                            NCCL_WIN_COLL_SYMMETRIC | NCCL_WIN_STRICT_ORDERING);
    if (window == nullptr) {
      // RegisterWindow() records the buffer even without a window; forget it
      // before the memory is freed below.
      windows_.erase(memory.ptr);
    }
    PADDLE_ENFORCE_NOT_NULL(
        window,
        common::errors::Unavailable(
            "ncclCommWindowRegister returned no window for the host memory; "
            "the communicator does not support symmetric memory."));
  } catch (...) {
    // Undo the steps that succeeded; errors here are ignored so that the
    // original exception propagates.
    auto address = reinterpret_cast<CUdeviceptr>(memory.ptr);
    if (is_mapped) {
      phi::dynload::cuMemUnmap(address, alloc_bytes);
    }
    if (memory.handle != 0) {
      phi::dynload::cuMemRelease(memory.handle);
    }
    if (address != 0) {
      phi::dynload::cuMemAddressFree(address, alloc_bytes);
    }
    throw;
  }
  device_api_->host_windows.emplace(window, memory);
  VLOG(3) << "Allocated NCCL host window " << window << " at " << memory.ptr
          << " of size " << alloc_bytes << " on host NUMA node " << numa_id;
  return NCCLHostWindow{window, memory.ptr, alloc_bytes};
#else
  PADDLE_THROW(common::errors::Unavailable(
      "Paddle was not built with PADDLE_WITH_GIN_OVERLAP (the NCCL device "
      "API: NCCL 2.30.7 or newer with nccl_device.h, CUDA 13.0 or newer, "
      "sm_90 or newer)."));
#endif
}

void NCCLCommContext::FreeHostWindow(void* window) {
#ifdef PADDLE_WITH_GIN_OVERLAP
  auto iter = device_api_->host_windows.find(window);
  PADDLE_ENFORCE_NE(iter,
                    device_api_->host_windows.end(),
                    common::errors::InvalidArgument(
                        "Window %p was not allocated by AllocHostWindow() on "
                        "this communicator.",
                        window));
  DeviceApiState::HostMemory memory = iter->second;
  device_api_->host_windows.erase(iter);
  DeregisterWindow(memory.ptr);
  ReleaseHostMemory(memory.handle, memory.ptr, memory.bytes);
#else
  PADDLE_THROW(common::errors::Unavailable(
      "Paddle was not built with PADDLE_WITH_GIN_OVERLAP (the NCCL device "
      "API: NCCL 2.30.7 or newer with nccl_device.h, CUDA 13.0 or newer, "
      "sm_90 or newer)."));
#endif
}

void NCCLCommContext::FreeAllHostWindowMemory() {
#ifdef PADDLE_WITH_GIN_OVERLAP
  for (const auto& window_and_memory : device_api_->host_windows) {
    const DeviceApiState::HostMemory& memory = window_and_memory.second;
    ReleaseHostMemory(memory.handle, memory.ptr, memory.bytes);
  }
  device_api_->host_windows.clear();
#endif
}

bool NCCLCommContext::IsAllToAllAvailable() const {
#if defined(PADDLE_WITH_NCCL) && NCCL_VERSION_CODE >= 23007
  return phi::dynload::ncclAlltoAll.IsValid();
#else
  return false;
#endif
}

void NCCLCommContext::AllToAll(DenseTensor* out_tensor,
                               const DenseTensor& in_tensor,
                               gpuStream_t stream) {
#if defined(PADDLE_WITH_NCCL) && NCCL_VERSION_CODE >= 23007
  CommStaticCheck::SameShape(*out_tensor,
                             in_tensor,
                             /*dst_rank*/ rank_,
                             /*cur_rank*/ rank_,
                             size_);
  NCCL_CHECK(phi::dynload::ncclAlltoAll(in_tensor.data(),
                                        out_tensor->data(),
                                        in_tensor.numel() / size_,
                                        ToNCCLDataType(in_tensor.type()),
                                        nccl_comm_,
                                        stream));
#else
  PADDLE_THROW(common::errors::Unavailable(
      "ncclAlltoAll is not available in the NCCL library Paddle was compiled "
      "against, IsAllToAllAvailable() must be checked first."));
#endif
}

int NCCLCommContext::GetNcclVersion() { return nccl_version_; }

ncclComm_t NCCLCommContext::GetNcclComm() { return nccl_comm_; }

gpuStream_t NCCLCommContext::GetStream() { return dev_ctx_->stream(); }

phi::GPUContext* NCCLCommContext::GetDevContext() { return dev_ctx_.get(); }

void NCCLCommContext::SetDevContext(
    std::unique_ptr<phi::GPUContext>&& dev_ctx) {
  dev_ctx_ = std::move(dev_ctx);
}

gpuEvent_t NCCLCommContext::GetComputeEvent() { return compute_event_.get(); }

void NCCLCommContext::SetComputeEvent(
    std::shared_ptr<std::remove_pointer<phi::gpuEvent_t>::type>&&
        compute_event) {
  compute_event_ = std::move(compute_event);
}

gpuEvent_t NCCLCommContext::GetCommEvent() { return comm_event_.get(); }

void NCCLCommContext::SetCommEvent(
    std::shared_ptr<std::remove_pointer<phi::gpuEvent_t>::type>&& comm_event) {
  comm_event_ = std::move(comm_event);
}

void NCCLCommContext::Broadcast(DenseTensor* out_tensor,
                                const DenseTensor& in_tensor,
                                int root,
                                gpuStream_t stream) {
  CommStaticCheck::SameShape(*out_tensor,
                             in_tensor,
                             /*dst_rank*/ rank_,
                             /*cur_rank*/ rank_,
                             size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    NCCLDynamicCheck::CheckShape(*out_tensor, root, rank_, nccl_comm_);
  }
  NCCL_CHECK(phi::dynload::ncclBroadcast(in_tensor.data(),
                                         out_tensor->data(),
                                         in_tensor.numel(),
                                         ToNCCLDataType(in_tensor.type()),
                                         root,
                                         nccl_comm_,
                                         stream));
}

void NCCLCommContext::AllGather(DenseTensor* out_tensor,
                                const DenseTensor& in_tensor,
                                gpuStream_t stream) {
  phi::distributed::CommStaticCheck::GatherLikeShape(*out_tensor,
                                                     in_tensor,
                                                     /*dst_rank*/ rank_,
                                                     /*cur_rank*/ rank_,
                                                     size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    phi::distributed::NCCLDynamicCheck::CheckShape(*out_tensor,
                                                   /*root_rank*/ 0,
                                                   rank_,
                                                   nccl_comm_);
  }
  NCCL_CHECK(phi::dynload::ncclAllGather(in_tensor.data(),
                                         out_tensor->data(),
                                         in_tensor.numel(),
                                         ToNCCLDataType(in_tensor.type()),
                                         nccl_comm_,
                                         stream));
}
void NCCLCommContext::ReduceScatter(DenseTensor* out_tensor,
                                    const DenseTensor& in_tensor,
                                    ncclRedOp_t reduce_type,
                                    gpuStream_t stream) {
  PADDLE_ENFORCE_EQ(
      in_tensor.dtype() != DataType::FLOAT8_E4M3FN &&
          in_tensor.dtype() != DataType::FLOAT8_E5M2,
      true,
      common::errors::InvalidArgument(
          "float8 dtypes are not currently supported for NCCL reductions"));

  phi::distributed::CommStaticCheck::ScatterLikeShape(*out_tensor,
                                                      in_tensor,
                                                      /*dst_rank*/ rank_,
                                                      /*cur_rank*/ rank_,
                                                      size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    phi::distributed::NCCLDynamicCheck::CheckShape(*out_tensor,
                                                   /*root_rank*/ 0,
                                                   rank_,
                                                   nccl_comm_);
  }
  NCCL_CHECK(phi::dynload::ncclReduceScatter(in_tensor.data(),
                                             out_tensor->data(),
                                             out_tensor->numel(),
                                             ToNCCLDataType(in_tensor.type()),
                                             reduce_type,
                                             nccl_comm_,
                                             stream));
}

void NCCLCommContext::Send(const DenseTensor& in_tensor,
                           const int64_t& count,
                           const int& peer,
                           gpuStream_t stream) {
  phi::distributed::CommStaticCheck::CheckShape(in_tensor, rank_, size_);

  if (FLAGS_enable_nccl_dynamic_check) {
    NCCLDynamicCheck::CheckShape(in_tensor, rank_, rank_, nccl_comm_);
  }

  NCCL_CHECK(phi::dynload::ncclSend(in_tensor.data(),
                                    count,
                                    ToNCCLDataType(in_tensor.dtype()),
                                    peer,
                                    nccl_comm_,
                                    stream));
  VLOG(3) << "rank " << GetRank() << " send "
          << common::product(in_tensor.dims()) << " to " << peer;
}

void NCCLCommContext::Recv(DenseTensor* out_tensor,
                           const int64_t& count,
                           const int& peer,
                           gpuStream_t stream) {
  phi::distributed::CommStaticCheck::CheckShape(*out_tensor, rank_, size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    NCCLDynamicCheck::CheckShape(*out_tensor, peer, rank_, nccl_comm_);
  }

  NCCL_CHECK(phi::dynload::ncclRecv(out_tensor->data(),
                                    count,
                                    ToNCCLDataType(out_tensor->dtype()),
                                    peer,
                                    nccl_comm_,
                                    stream));
  VLOG(3) << "rank " << GetRank() << " recv "
          << common::product(out_tensor->dims()) << " from " << peer;
}

void NCCLCommContext::AllReduce(DenseTensor* out_tensor,
                                const DenseTensor& in_tensor,
                                ncclRedOp_t reduce_type,
                                gpuStream_t stream) {
  PADDLE_ENFORCE_EQ(
      in_tensor.dtype() != DataType::FLOAT8_E4M3FN &&
          in_tensor.dtype() != DataType::FLOAT8_E5M2,
      true,
      common::errors::InvalidArgument(
          "float8 dtypes are not currently supported for NCCL reductions"));

  phi::distributed::CommStaticCheck::SameShape(*out_tensor,
                                               in_tensor,
                                               /*dst_rank*/ rank_,
                                               /*cur_rank*/ rank_,
                                               size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    phi::distributed::NCCLDynamicCheck::CheckShape(*out_tensor,
                                                   /*root_rank*/ 0,
                                                   rank_,
                                                   nccl_comm_);
  }
  NCCL_CHECK(phi::dynload::ncclAllReduce(in_tensor.data(),
                                         out_tensor->data(),
                                         in_tensor.numel(),
                                         ToNCCLDataType(in_tensor.type()),
                                         reduce_type,
                                         nccl_comm_,
                                         stream));
}

void NCCLCommContext::Reduce(DenseTensor* out_tensor,
                             const DenseTensor& in_tensor,
                             ncclRedOp_t reduce_type,
                             int root,
                             gpuStream_t stream) {
  PADDLE_ENFORCE_EQ(
      in_tensor.dtype() != DataType::FLOAT8_E4M3FN &&
          in_tensor.dtype() != DataType::FLOAT8_E5M2,
      true,
      common::errors::InvalidArgument(
          "float8 dtypes are not currently supported for NCCL reductions"));

  phi::distributed::CommStaticCheck::SameShape(*out_tensor,
                                               in_tensor,
                                               /*dst_rank*/ root,
                                               /*cur_rank*/ rank_,
                                               size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    phi::distributed::NCCLDynamicCheck::CheckShape(*out_tensor,
                                                   /*root_rank*/ root,
                                                   rank_,
                                                   nccl_comm_);
  }
  NCCL_CHECK(phi::dynload::ncclReduce(in_tensor.data(),
                                      out_tensor->data(),
                                      in_tensor.numel(),
                                      ToNCCLDataType(in_tensor.type()),
                                      reduce_type,
                                      root,
                                      nccl_comm_,
                                      stream));
}

void NCCLCommContext::GroupStart() {
  NCCL_CHECK(phi::dynload::ncclGroupStart());
}
void NCCLCommContext::GroupEnd() { NCCL_CHECK(phi::dynload::ncclGroupEnd()); }

#if NCCL_VERSION_CODE >= 21100
void NCCLCommContext::RedOpCreatePreMulSum(ncclRedOp_t* op,
                                           void* scalar,
                                           ncclDataType_t dtype,
                                           ncclScalarResidence_t residence) {
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclRedOpCreatePreMulSum(
      op, scalar, dtype, residence, nccl_comm_));
}

void NCCLCommContext::RedOpDestroy(ncclRedOp_t op) {
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclRedOpDestroy(op, nccl_comm_));
}
#endif

}  // namespace phi::distributed
