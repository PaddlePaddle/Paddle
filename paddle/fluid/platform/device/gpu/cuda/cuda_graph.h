// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "cuda.h"          // NOLINT
#include "cuda_runtime.h"  // NOLINT
#include "paddle/fluid/platform/device/gpu/gpu_types.h"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace platform {

#if CUDA_VERSION >= 10010
static void ThrowErrorIfNotSupportCUDAGraph() {}
#else
enum cudaStreamCaptureMode {
  cudaStreamCaptureModeGlobal = 0,
  cudaStreamCaptureModeThreadLocal = 1,
  cudaStreamCaptureModeRelaxed = 2
};
static void ThrowErrorIfNotSupportCUDAGraph() {
  PADDLE_THROW(platform::errors::Unimplemented(
      "CUDA Graph is only supported when CUDA version >= 10.1"));
}
#endif

// NOTE: Currently, we do not support to capture CUDA graph in parallel
// NOTE: Do not use this class directly because it should be used with
//       the memory pool.
class CUDAGraph {
  DISABLE_COPY_AND_ASSIGN(CUDAGraph);

  // Since the constructor would throw error is CUDA_VERSION < 10010.
  // The non-static method of CUDAGraph need not check CUDA_VERSION
  // again.
  CUDAGraph() {
    ThrowErrorIfNotSupportCUDAGraph();
    id_ = UniqueID();
  }

 public:
  ~CUDAGraph() { Reset(); }

  CUDAGraphID ID() const { return id_; }

  void Replay();

  void Reset();

  void AddResetCallback(std::function<void()> callback) {
    std::lock_guard<std::mutex> guard(mtx_);
    callbacks_.push_back(std::move(callback));
  }

  void PrintToDotFiles(const std::string &dirname, unsigned int flags);

  static void BeginCapture(platform::CUDAPlace place, cudaStream_t stream,
                           cudaStreamCaptureMode mode);
  static std::unique_ptr<CUDAGraph> EndCapture();

  static void BeginSegmentCapture();
  static void EndSegmentCapture();

  static void AddResetCallbackDuringCapturing(std::function<void()> callback) {
    capturing_graph_->AddResetCallback(std::move(callback));
  }

  // No need to add CUDA_VERSION macro because capturing_graph_ would
  // always be nullptr (constructor throws error)
  static bool IsCapturing() { return capturing_graph_ != nullptr; }

  static CUDAGraphID CapturingID() { return capturing_graph_->id_; }

  static platform::CUDAPlace CapturingPlace() {
    return capturing_graph_->place_;
  }

  // This API can be used to debug which GPU operation is not
  // supported during capturing CUDA Graph.
  static bool IsValidCapturing();

  static bool IsThreadLocalCapturing() {
#if CUDA_VERSION >= 10010
    return IsCapturing() &&
           capturing_graph_->capture_mode_ == cudaStreamCaptureModeThreadLocal;
#else
    return false;
#endif
  }

  static bool IsThisThreadCapturing() {
    if (UNLIKELY(IsCapturing())) {
      return IsThreadLocalCapturing()
                 ? capturing_thread_id_.get() == std::this_thread::get_id()
                 : true;
    } else {
      return false;
    }
  }

 private:
  static CUDAGraphID UniqueID() {
    static std::atomic<CUDAGraphID> id;
    return id.fetch_add(1);
  }

 private:
#if CUDA_VERSION >= 10010
  std::vector<cudaGraph_t> graphs_;
  std::vector<cudaGraphExec_t> exec_graphs_;
  cudaStreamCaptureMode capture_mode_;
#endif
  cudaStream_t stream_{nullptr};
  platform::CUDAPlace place_;
  CUDAGraphID id_;
  std::vector<std::function<void()>> callbacks_;
  bool is_reset_{false};
  std::mutex mtx_;

  static paddle::optional<std::thread::id> capturing_thread_id_;
  static std::unique_ptr<CUDAGraph> capturing_graph_;
};

#if CUDA_VERSION >= 10010
class CUDAGraphCaptureModeGuard {
  DISABLE_COPY_AND_ASSIGN(CUDAGraphCaptureModeGuard);

 public:
  explicit CUDAGraphCaptureModeGuard(
      cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed) {
    if (UNLIKELY(CUDAGraph::IsCapturing())) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaThreadExchangeStreamCaptureMode(&mode));
      // After cudaThreadExchangeStreamCaptureMode is called,
      // the variable "mode" would be set to the old capturing mode.
      old_mode_ = mode;
    }
  }

  ~CUDAGraphCaptureModeGuard() PADDLE_MAY_THROW {
    if (UNLIKELY(CUDAGraph::IsCapturing())) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaThreadExchangeStreamCaptureMode(&old_mode_));
    }
  }

 private:
  cudaStreamCaptureMode old_mode_;
};
#else
class CUDAGraphCaptureModeGuard {
  DISABLE_COPY_AND_ASSIGN(CUDAGraphCaptureModeGuard);

 public:
  explicit CUDAGraphCaptureModeGuard(
      cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed) {}
};
#endif

}  // namespace platform
}  // namespace paddle
