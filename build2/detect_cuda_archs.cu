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

#include "cuda.h"
#include "cuda_runtime.h"
#include "stdio.h"
int main() {
  int count = 0;
  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;
  if (count == 0) return -1;
  for (int device = 0; device < count; ++device) {
    cudaDeviceProp prop;
    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))
      printf("%d.%d ", prop.major, prop.minor);
  }
  return 0;
}
