/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <thrust/sort.h>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

#include "paddle/fluid/operators/detection/nms_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/phi/kernels/funcs/slice.h"
#include "paddle/phi/kernels/gpu/reduce.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"

namespace paddle {
namespace operators {

template <typename T>
struct AttnMaskFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    return b >= 0 ? a : 0;
  }
};

__global__ void FillIndex(int* indices, int num_rows, int num_cols) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_rows * num_cols) return;

  int col = tid % num_cols;
  int row = tid / num_cols;

  // for (int j = row; j < num_rows; j += num_cols) {
  //     for (int i = col; i < num_cols; i += num_rows) {
  //     indices[j * num_cols + i] = i;
  //     }
  // }
  indices[tid] = col;
}

template <typename T>
__global__ void SlicedArgsort(T* data, int* indices, int num_rows,
                              int num_cols) {
  auto raw = blockIdx.x * blockDim.x + threadIdx.x;
  if (raw >= num_rows) return;
  thrust::sort_by_key(thrust::seq, data + raw * num_cols + 1,
                      data + (raw + 1) * num_cols, indices + raw * num_cols + 1,
                      thrust::greater<T>());
}

template <typename T>
__global__ void TakeAlongAxis(const T* src, T* dst, int* indices, int num_rows,
                              int src_num_cols, int dst_num_cols,
                              int num_elements) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_rows * dst_num_cols) return;

  int raw = tid / dst_num_cols;
  int col = tid % dst_num_cols;
  for (int i = 0; i < num_elements; ++i) {
    dst[tid * num_elements + i] =
        *(src + (raw * src_num_cols + indices[tid]) * num_elements + i);
  }
}

inline int ComputeBlockSize(int col) {
  if (col > 512)
    return 1024;
  else if (col > 256 && col <= 512)
    return 512;
  else if (col > 128 && col <= 256)
    return 256;
  else if (col > 64 && col <= 128)
    return 128;
  else
    return 64;
}

}  // namespace operators
}  // namespace paddle
