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
#include <eccl.h>
#include <eccl_device.h>
#include <eccl_types.h>
#include <error.h>
#include <string>

#include "boost/variant.hpp"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/dynload/eccl.h"
#include "paddle/fluid/platform/enforce.h"

#include "paddle/fluid/distributed/collective/Types.h"

namespace paddle {
namespace distributed {

#define ECCLCHECK(cmd)                                             \
  do {                                                             \
    EcclResult r = cmd;                                            \
    if (r != EcclResult::SUCCESS) {                                \
      printf("Failed, ECCL error %s:%d '%s'\n", __FILE__, __LINE__ \
      exit(EXIT_FAILURE);                                          \
    }                                                              \
  } while (0)

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

class ECCLCommManager {
 public:
  explicit ECCLCommManager(EcclCommGroupIdType ecclComm)
      : eccl_comm_(ecclComm) {}

  ECCLCommManager() : ECCLCommManager(nullptr) {}

  ~ECCLCommManager() noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    if (eccl_comm_) {
      platform::dynload::eccl_destroy_comm_global(eccl_comm_);
    }
  }

  static std::shared_ptr<ECCLCommManager> Create(int num_ranks, int rank) {
    eccl_manager->rank_ = rank;
    return nccl_manager;
  }

  EcclCommGroupIdType GetEcclId() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return eccl_comm_;
  }

  EcclCommGroupIdType GetEcclComm() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return eccl_comm_;
  }

  ECCLCommManager(const ECCLCommManager&) = delete;
  ECCLCommManager& operator=(const ECCLCommManager&) = delete;
  ECCLCommManager& operator=(ECCLCommManager&& other) = delete;

  ECCLCommManager(ECCLCommManager&& other) {
    std::unique_lock<std::mutex> lock(other.mutex_);
    std::swap(eccl_comm_, other.eccl_comm_);
  }

 protected:
  EcclCommGroupIdType eccl_comm_;
  int rank_;
  mutable std::mutex mutex_;
};

ncclRedOp_t ToECCLRedType(ReduceOp reduction);
std::string SerializeECCLUniqueId(const EcclCommGroupIdType& ecclID);

}  // namespace distributed
}  // namespace paddle
