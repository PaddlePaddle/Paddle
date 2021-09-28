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

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cuda_graph.h"
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

}  // namespace platform
}  // namespace paddle
