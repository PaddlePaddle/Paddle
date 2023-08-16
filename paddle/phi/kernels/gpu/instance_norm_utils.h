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

#include <algorithm>
#include <cfloat>
#include <string>
#include <vector>
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/common/amp_type_traits.h"

namespace phi {

template <typename T>
using CudnnDataType = phi::backends::gpu::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename T>
static __global__ void repeat_param(const T *input,
                                    T *output,
                                    const int repeat_num,
                                    const int C) {
  CUDA_KERNEL_LOOP(i, repeat_num * C) {
    int index = i % C;
    output[i] = input[index];
  }
}

template <typename T, int BlockDim, bool AVG>
static __global__ void add_param(const T *input,
                                 T *output,
                                 const int repeat_num,
                                 const int C) {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  typedef cub::BlockReduce<MPType, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ou_storage;
  for (int i = blockIdx.x; i < C; i += gridDim.x) {
    MPType ou = static_cast<MPType>(0);
    for (int j = threadIdx.x; j < repeat_num; j += blockDim.x) {
      const int index = j * C + i;
      ou = ou + static_cast<MPType>(input[index]);
    }
    ou = BlockReduce(ou_storage).Reduce(ou, cub::Sum());
    if (threadIdx.x == 0) {
      output[i] = static_cast<T>(ou);
    }
    __syncthreads();

    if (AVG) {
      output[i] = static_cast<T>(static_cast<MPType>(output[i]) / repeat_num);
    }
  }
}
}  // namespace phi
