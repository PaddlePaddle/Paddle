// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/common/macros.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

// Print tensor dtype, shape, and all data values using a single CUDA thread.
// The kernel is submitted to `stream` so it is ordered after any preceding
// operations on that stream.  Output is flushed via cudaStreamSynchronize
// before this function returns.
// Only GPU DenseTensors are supported.
PADDLE_API void DebugPrintGPUTensor(const phi::DenseTensor& tensor,
                                    cudaStream_t stream);

}  // namespace phi
