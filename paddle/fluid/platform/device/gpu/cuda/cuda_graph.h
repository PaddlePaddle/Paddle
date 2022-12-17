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

#include "paddle/phi/backends/gpu/cuda/cuda_graph.h"

namespace paddle {
namespace platform {

using CUDAKernelParams = phi::backends::gpu::CUDAKernelParams;
#if CUDA_VERSION < 10010
using cudaStreamCaptureMode = phi::backends::gpu::cudaStreamCaptureMode;
#endif
using CUDAGraph = phi::backends::gpu::CUDAGraph;
using CUDAGraphCaptureModeGuard = phi::backends::gpu::CUDAGraphCaptureModeGuard;
using IsSameKernelHelper = phi::backends::gpu::IsSameKernelHelper;

}  // namespace platform
}  // namespace paddle
