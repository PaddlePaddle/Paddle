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

#pragma once

#include <cuda_runtime_api.h>

#include <c10/core/Device.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <memory>
#include <optional>

namespace at::cuda {

/**
 * CUDAEvent is a movable not copyable wrapper around CUDA events.
 * This provides compatibility with PyTorch's CUDAEvent API.
 */
struct CUDAEvent {
  // Default constructor
  CUDAEvent() noexcept = default;

  // Constructor with flags (cudaEventDisableTiming, etc.)
  explicit CUDAEvent(unsigned int flags) noexcept : flags_(flags) {}

  // Destructor
  ~CUDAEvent() {
    if (is_created_) {
      cudaEventDestroy(event_);
    }
  }

  // Non-copyable
  CUDAEvent(const CUDAEvent&) = delete;
  CUDAEvent& operator=(const CUDAEvent&) = delete;

  // Movable
  CUDAEvent(CUDAEvent&& other) noexcept { moveHelper(std::move(other)); }
  CUDAEvent& operator=(CUDAEvent&& other) noexcept {
    if (this != &other) {
      moveHelper(std::move(other));
    }
    return *this;
  }

  // Implicit conversion to cudaEvent_t
  operator cudaEvent_t() const { return event(); }

  // Get the underlying cuda event
  cudaEvent_t event() const { return event_; }

  // Check if event has been created
  bool isCreated() const { return is_created_; }

  // Get device index
  c10::DeviceIndex device_index() const { return device_index_; }

  // Query if event is completed
  bool query() const {
    if (!is_created_) {
      return true;
    }
    cudaError_t err = cudaEventQuery(event_);
    if (err == cudaSuccess) {
      return true;
    } else if (err != cudaErrorNotReady) {
      C10_CUDA_CHECK(err);
    }
    return false;
  }

  // Record event on current stream
  void record() { record(getCurrentCUDAStream()); }

  // Record event on specified stream
  void record(const CUDAStream& stream) {
    if (!is_created_) {
      createEvent(stream.unwrap().device_index());
    }
    TORCH_CHECK(device_index_ == stream.unwrap().device_index(),
                "Event device ",
                device_index_,
                " does not match recording stream's device ",
                stream.unwrap().device_index(),
                ".");
    c10::cuda::CUDAGuard guard(device_index_);
    C10_CUDA_CHECK(cudaEventRecord(event_, stream.stream()));
  }

  // Record event once (only if not already recorded)
  void recordOnce(const CUDAStream& stream) {
    if (!was_recorded_) {
      record(stream);
      was_recorded_ = true;
    }
  }

  // Make a stream wait for this event
  void block(const CUDAStream& stream) {
    if (is_created_) {
      c10::cuda::CUDAGuard guard(stream.unwrap().device_index());
      C10_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), event_, 0));
    }
  }

  // Synchronize on this event
  void synchronize() const {
    if (is_created_) {
      C10_CUDA_CHECK(cudaEventSynchronize(event_));
    }
  }

  // Get elapsed time between two events
  float elapsed_time(const CUDAEvent& other) const {
    TORCH_CHECK(
        is_created_ && other.isCreated(),
        "Both events must be recorded before calculating elapsed time.");
    TORCH_CHECK(
        query() && other.query(),
        "Both events must be completed before calculating elapsed time.");
    float time_ms = 0;
    c10::cuda::CUDAGuard guard(device_index_);
    C10_CUDA_CHECK(cudaEventElapsedTime(&time_ms, event_, other.event_));
    return time_ms;
  }

 private:
  unsigned int flags_ = cudaEventDisableTiming;
  bool is_created_ = false;
  bool was_recorded_ = false;
  c10::DeviceIndex device_index_ = -1;
  cudaEvent_t event_{};

  void createEvent(c10::DeviceIndex device_index) {
    device_index_ = device_index;
    c10::cuda::CUDAGuard guard(device_index_);
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags_));
    is_created_ = true;
  }

  void moveHelper(CUDAEvent&& other) {
    flags_ = other.flags_;
    is_created_ = other.is_created_;
    was_recorded_ = other.was_recorded_;
    device_index_ = other.device_index_;
    event_ = other.event_;
    other.is_created_ = false;
    other.event_ = cudaEvent_t{};
  }
};

}  // namespace at::cuda

// torch::Event alias for compatibility
namespace torch {
using at::cuda::CUDAEvent;
using at::cuda::CUDAStream;
}  // namespace torch
