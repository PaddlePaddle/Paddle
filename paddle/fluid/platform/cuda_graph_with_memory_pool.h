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

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/device/gpu/cuda/cuda_graph.h"
#endif

namespace paddle {
namespace platform {

// NOTE: These APIs are not thread-safe.
#ifdef PADDLE_WITH_CUDA
void BeginCUDAGraphCapture(platform::CUDAPlace place,
                           cudaStreamCaptureMode mode);
std::unique_ptr<CUDAGraph> EndCUDAGraphCapture();
#endif

inline bool IsCUDAGraphCapturing() {
#ifdef PADDLE_WITH_CUDA
  return CUDAGraph::IsCapturing();
#else
  return false;
#endif
}

inline platform::CUDAPlace CUDAGraphCapturingPlace() {
#ifdef PADDLE_WITH_CUDA
  return CUDAGraph::CapturingPlace();
#else
  PADDLE_THROW(platform::errors::Unimplemented(
      "CUDA Graph is only supported on NVIDIA GPU device."));
#endif
}

// Add reset callback if CUDA Graph is capturing.
// Otherwise, invoke callback directly.
template <typename Callback>
inline void AddResetCallbackIfCapturingCUDAGraph(Callback &&callback) {
#ifdef PADDLE_WITH_CUDA
  if (UNLIKELY(IsCUDAGraphCapturing())) {
    return CUDAGraph::AddResetCallbackDuringCapturing(
        std::forward<Callback>(callback));
  }
#endif
  callback();
}

template <typename T>
inline T *RestoreHostMemIfCapturingCUDAGraph(T *host_mem, size_t size) {
  static_assert(std::is_trivial<T>::value, "T must be trivial type");
  static_assert(!std::is_same<T, void>::value, "T cannot be void");
#ifdef PADDLE_WITH_CUDA
  if (UNLIKELY(IsCUDAGraphCapturing())) {
    size_t nbytes = size * sizeof(T);
    void *new_host_mem = new uint8_t[nbytes];
    std::memcpy(new_host_mem, host_mem, nbytes);
    AddResetCallbackIfCapturingCUDAGraph(
        [new_host_mem] { delete[] reinterpret_cast<uint8_t *>(new_host_mem); });
    return reinterpret_cast<T *>(new_host_mem);
  }
#endif
  return host_mem;
}

class SkipCUDAGraphCaptureGuard {
  DISABLE_COPY_AND_ASSIGN(SkipCUDAGraphCaptureGuard);

 public:
  SkipCUDAGraphCaptureGuard() {
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10010
    if (UNLIKELY(CUDAGraph::IsCapturing())) {
      CUDAGraph::EndSegmentCapture();
    }
#endif
#endif
  }

  ~SkipCUDAGraphCaptureGuard() {
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10010
    if (UNLIKELY(CUDAGraph::IsCapturing())) {
      CUDAGraph::BeginSegmentCapture();
    }
#endif
#endif
  }
};

}  // namespace platform
}  // namespace paddle
