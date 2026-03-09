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

namespace phi {
namespace funcs {

// Cast index tensor elements from type T (e.g. int32) to int64_t on GPU.
// This is needed because the index_fill kernel always works with int64 indices
// internally, but users may pass int32 index tensors.
template <typename T>
__global__ void CastToInt64Kernel(const T* input,
                                  int64_t* output,
                                  int64_t numel) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
      static_cast<int64_t>(threadIdx.x);
  if (idx < numel) {
    output[idx] = static_cast<int64_t>(input[idx]);
  }
}

}  // namespace funcs
}  // namespace phi
