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

// scale.h: in-place scale kernel on one GPU.
//   - The launch function returns cudaGetLastError() after the launch.
//   - Block and thread counts do not change any result bit: they only decide
//     which elements a thread handles.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace paddle {
namespace distributed {

// data[i] = __fmul_rn(data[i], scale) for i in [0, numel), in place.
// <<<num_blocks, num_threads>>>
cudaError_t LaunchScaleRange(float* data,
                             size_t numel,
                             float scale,
                             int num_blocks,
                             cudaStream_t stream,
                             int num_threads = 256);

}  // namespace distributed
}  // namespace paddle
