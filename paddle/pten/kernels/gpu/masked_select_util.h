// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

__global__ void SetMaskArray(const bool* mask, int32_t* mask_array, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < size; idx += blockDim.x * gridDim.x) {
    if (mask[idx])
      mask_array[idx] = 1;
    else
      mask_array[idx] = 0;
  }
}
