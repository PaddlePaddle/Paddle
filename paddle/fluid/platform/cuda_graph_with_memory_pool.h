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

#include "paddle/common/macros.h"
#include "paddle/phi/backends/gpu/cuda/cuda_graph_with_memory_pool.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace platform {

// NOTE: These APIs are not thread-safe.
#ifdef PADDLE_WITH_CUDA
using CUDAGraph = phi::backends::gpu::CUDAGraph;

void BeginCUDAGraphCapture(phi::GPUPlace place,
                           cudaStreamCaptureMode mode,
                           int64_t pool_id = CUDAGraph::kInvalidPoolID);
std::unique_ptr<CUDAGraph> EndCUDAGraphCapture();
#endif

inline phi::GPUPlace CUDAGraphCapturingPlace() {
#ifdef PADDLE_WITH_CUDA
  return CUDAGraph::CapturingPlace();
#else
  PADDLE_THROW(phi::errors::Unimplemented(
      "CUDA Graph is only supported on NVIDIA GPU device."));
#endif
}

using phi::backends::gpu::IsCUDAGraphCapturing;

using phi::backends::gpu::AddResetCallbackIfCapturingCUDAGraph;

using phi::backends::gpu::RestoreHostMemIfCapturingCUDAGraph;

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
