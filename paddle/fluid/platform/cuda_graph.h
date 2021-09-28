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

#include <functional>
#include <memory>
#include "cuda.h"          // NOLINT
#include "cuda_runtime.h"  // NOLINT
#include "paddle/fluid/platform/type_defs.h"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace platform {

#if CUDA_VERSION >= 10010
static void ThrowErrorIfNotSupportCUDAGraph() {}
#else
static void ThrowErrorIfNotSupportCUDAGraph() {
  PADDLE_THROW(errors::Unimplemented(
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
  CUDAGraph() { ThrowErrorIfNotSupportCUDAGraph(); }

 public:
  ~CUDAGraph() { Reset(); }

  CUDAGraphID ID() const { return id_; }

  void Replay();

  void Reset();

  void SetResetCallback(const std::function<void()> &callback) {
    callback_ = callback;
  }

  static void BeginCapture(platform::CUDAPlace place, cudaStream_t stream,
                           cudaStreamCaptureMode mode);
  static std::unique_ptr<CUDAGraph> EndCapture();

  // No need to add CUDA_VERSION macro because capturing_graph_ would
  // always be nullptr (constructor throws error)
  static bool IsCapturing() { return capturing_graph_ != nullptr; }

  static CUDAGraphID CapturingID() { return capturing_graph_->id_; }

 private:
#if CUDA_VERSION >= 10010
  cudaGraph_t graph_{nullptr};
  cudaGraphExec_t exec_graph_{nullptr};
#endif
  cudaStream_t stream_{nullptr};
  platform::CUDAPlace place_;
  CUDAGraphID id_{0};
  std::function<void()> callback_;
  bool is_reset_{false};

  static std::unique_ptr<CUDAGraph> capturing_graph_;
};

#if CUDA_VERSION >= 10010
class CUDAGraphCaptureModeGuard {
  DISABLE_COPY_AND_ASSIGN(CUDAGraphCaptureModeGuard);

 public:
  explicit CUDAGraphCaptureModeGuard(cudaStreamCaptureMode new_mode) {
    old_mode_ = new_mode;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaThreadExchangeStreamCaptureMode(&old_mode_));
  }

  ~CUDAGraphCaptureModeGuard() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaThreadExchangeStreamCaptureMode(&old_mode_));
  }

 private:
  cudaStreamCaptureMode old_mode_;
};
#else
class CUDAGraphCaptureModeGuard {
  DISABLE_COPY_AND_ASSIGN(CUDAGraphCaptureModeGuard);

 public:
  CUDAGraphCaptureModeGuard() = default;
};
#endif

}  // namespace platform
}  // namespace paddle
