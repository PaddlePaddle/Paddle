/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include<cuda.h>
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

template<class T>
__global__ void sum_gpu(const T *in_0, const T *in_1,  T* out, int64_t N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
    while (id < N) {
      out[id] = in_0[id] + in_1[id];
      id += blockDim.x * gridDim.x;
    }
}

template<class T>
__global__ void sum_gpu_array(T **in,  T* out, int64_t N, size_t in_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < N) {
    T total = 0;
    for (int i = 0; i < in_size; ++i) {
      const T *tmp = in[i];
      if (tmp != nullptr)
        total += tmp[id];
      }
      out[id] = total;
      id += blockDim.x * gridDim.x;
    }
}

template<class T>
__global__ void sum_gpu_sr(T **sr_in, T** sr_out, int64_t N, size_t rows) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < N) {
    T total = 0;
    for (int i = 0; i < rows; ++i) {
      const T *tmp = sr_in[i];
      T *tmp_out = sr_out[i];
      if (tmp != nullptr && tmp_out != nullptr) {
        total += tmp[id];
      }
      tmp_out[id] = total;
    }
    id += blockDim.x * gridDim.x;
  }
}

template<class T>
__global__ void sum_gpu4(const T *in_0, const T *in_1, T *out, int64_t N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = id; i < N / 4; i += blockDim.x * gridDim.x) {
    float4 *in0_4 = reinterpret_cast<float4*> (const_cast<T*>(in_0));
    float4 *in1_4 = reinterpret_cast<float4*> (const_cast<T*>(in_1));
    float4 tmp;
    tmp.x = in0_4[i].x + in1_4[i].x;
    tmp.y = in0_4[i].y + in1_4[i].y;
    tmp.z = in0_4[i].z + in1_4[i].z;
    tmp.w = in0_4[i].w + in1_4[i].w;
    reinterpret_cast<float4*>(out)[i] = tmp;
  }
}
}
}
