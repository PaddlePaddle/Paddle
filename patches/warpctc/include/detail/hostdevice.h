// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef __CUDACC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

// NOTE(dzhwinter)
// the warp primitive is different in cuda9(Volta) GPU.
// add a wrapper to compatible with cuda7 to cuda9
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define DEFAULT_MASK 0u
template <typename T>
__forceinline__ __device__ T __shfl_down(T input, int delta) {
  return __shfl_down_sync(DEFAULT_MASK, input, delta);
}

template <typename T>
__forceinline__ __device__ T __shfl_up(T input, int delta) {
  return __shfl_up_sync(DEFAULT_MASK, input, delta);
}

#endif
