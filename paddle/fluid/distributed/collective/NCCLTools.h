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

#include <cuda_runtime.h>
#include <error.h>
#include <string>

#include "boost/variant.hpp"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

#define NCCLCHECK(cmd)                                              \
  do {                                                              \
    ncclResult_t r = cmd;                                           \
    if (r != ncclSuccess) {                                         \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
             platform::dynload::ncclGetErrorString(r));             \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

// NOTE(shenliang03): EventManager are movable not copyable CudaEvent wrapper.
// EventManage is different from paddle::platform::CudaEvent.
// It uses lazy initialization and is only created when the
// Record() method is called for the first time; it also monitors
// device information to ensure that recorded stream and event
// are on the same device.

class EventManager {
 public:
  EventManager() {}
  explicit EventManager(unsigned int flags) : flags_{flags} {}

  ~EventManager() {
    if (is_created_) {
      platform::CUDADeviceGuard guard(device_index_);
      cudaEventDestroy(event_);
    }
  }

  EventManager(const EventManager&) = delete;
  EventManager& operator=(const EventManager&) = delete;

  EventManager(EventManager&& other) {
    std::swap(flags_, other.flags_);
    std::swap(is_created_, other.is_created_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }

  EventManager& operator=(EventManager&& other) {
    std::swap(flags_, other.flags_);
    std::swap(is_created_, other.is_created_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
    return *this;
  }

  bool IsCreated() const { return is_created_; }
  bool DeviceId() const { return device_index_; }
  gpuEvent_t GetRawCudaEvent() const { return event_; }

  void Record(const paddle::platform::CUDADeviceContext& ctx) {
    auto device_index = ctx.GetPlace().device;
    if (!is_created_) {
      CreateEvent(device_index);
    }
    PADDLE_ENFORCE_EQ(device_index, device_index_,
                      platform::errors::PreconditionNotMet(
                          "CUDADeviceContext's device %d does not match"
                          "Event's device %d",
                          device_index, device_index_));

    platform::CUDADeviceGuard guard(device_index_);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event_, ctx.stream()));
  }

  bool Query() const {
    gpuError_t err = cudaEventQuery(event_);
    if (err == cudaSuccess) {
      return true;
    } else if (err == cudaErrorNotReady) {
      return false;
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(err);
      return false;
    }
  }

  void Synchronize() const {
    if (is_created_) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaEventSynchronize(event_));
    }
  }

  void Block(const paddle::platform::CUDADeviceContext& ctx) const {
    if (is_created_) {
      auto device_index = ctx.GetPlace().device;
      PADDLE_ENFORCE_EQ(device_index, device_index_,
                        platform::errors::PreconditionNotMet(
                            "CUDADeviceContext's device %d does not match"
                            "Event's device %d",
                            device_index, device_index_));
      platform::CUDADeviceGuard guard(device_index_);
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamWaitEvent(ctx.stream(), event_, 0));
    }
  }

 private:
  unsigned int flags_ = cudaEventDefault;
  bool is_created_{false};
  gpuEvent_t event_{};
  int8_t device_index_{0};

 private:
  void CreateEvent(int device_index) {
    device_index_ = device_index;
    platform::CUDADeviceGuard guard(device_index);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreateWithFlags(&event_, flags_));
    is_created_ = true;
  }
};

// NOTE(shenliang03): NCCLCommManager is more lightweight than
// platform::NCCLComm

class NCCLCommManager {
 public:
  explicit NCCLCommManager(ncclComm_t ncclComm) : nccl_comm_(ncclComm) {}

  NCCLCommManager() : NCCLCommManager(nullptr) {}

  ~NCCLCommManager() noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    if (nccl_comm_) {
      platform::dynload::ncclCommDestroy(nccl_comm_);
    }
  }

  static std::shared_ptr<NCCLCommManager> Create(int num_ranks, int rank,
                                                 ncclUniqueId comm_id) {
    auto nccl_manager = std::make_shared<NCCLCommManager>();
    NCCLCHECK(platform::dynload::ncclCommInitRank(&(nccl_manager->nccl_comm_),
                                                  num_ranks, comm_id, rank));

    nccl_manager->nccl_id_ = comm_id;
    nccl_manager->rank_ = rank;
    return nccl_manager;
  }

  ncclUniqueId GetNcclId() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return nccl_id_;
  }

  ncclComm_t GetNcclComm() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return nccl_comm_;
  }

  NCCLCommManager(const NCCLCommManager&) = delete;
  NCCLCommManager& operator=(const NCCLCommManager&) = delete;
  NCCLCommManager& operator=(NCCLCommManager&& other) = delete;

  NCCLCommManager(NCCLCommManager&& other) {
    std::unique_lock<std::mutex> lock(other.mutex_);
    std::swap(nccl_comm_, other.nccl_comm_);
  }

 protected:
  ncclComm_t nccl_comm_;
  ncclUniqueId nccl_id_;
  int rank_;
  mutable std::mutex mutex_;
};

}  // namespace distributed
}  // namespace paddle
