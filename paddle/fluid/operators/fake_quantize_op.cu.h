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

#ifndef PADDLE_FLUID_OPERATORS_FAKE_QUANTIZE_OP_CU_H_
#define PADDLE_FLUID_OPERATORS_FAKE_QUANTIZE_OP_CU_H_
#endif  // PADDLE_FLUID_OPERATORS_FAKE_QUANTIZE_OP_CU_H_

#include <string>

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/fake_quantize_op.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

template <typename T>
struct QuantizeDataType {
  using type = T;
};

template <>
struct QuantizeDataType<phi::dtype::float16> {
  using type = float;
};

template <typename T>
__global__ void FindAbsMaxKernel(const T *in, const int n, T *out) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  extern __shared__ char *shared_max_data_tmp[];
  auto shared_max_data = reinterpret_cast<T *>(shared_max_data_tmp);
  if (gridDim.x > 1) {
    T local_max_data = T(0);
    for (int i = bid; i < n; i += blockDim.x * gridDim.x) {
      T tmp = abs(in[i]);
      if (tmp > local_max_data) {
        local_max_data = tmp;
      }
    }
    shared_max_data[tid] = local_max_data;
  } else {
    if (bid < n) {
      shared_max_data[tid] = abs(in[bid]);
    } else {
      shared_max_data[tid] = T(0);
    }
  }
  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i && (shared_max_data[tid] < shared_max_data[tid + i])) {
      shared_max_data[tid] = shared_max_data[tid + i];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = shared_max_data[0];
  }
}

template <typename T>
__global__ void ClipAndQuantKernel(const T *in,
                                   const T *scale,
                                   const int qmax,
                                   const int round_type,
                                   const int n,
                                   T *out) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  using ComputeDataType = typename QuantizeDataType<T>::type;

  ComputeDataType s = static_cast<ComputeDataType>(scale[0]);
  ComputeDataType inv_s = phi::funcs::inverse(s);
  ComputeDataType qmax_t = static_cast<ComputeDataType>(qmax);

  for (int i = bid; i < n; i += blockDim.x * gridDim.x) {
    ComputeDataType x = static_cast<ComputeDataType>(in[i]);
    if (round_type == 0) {
      x = qmax_t * inv_s * x;
      if (qmax_t == static_cast<ComputeDataType>(448)) {
        x = float8_e4m3fn(x);
      } else if (qmax_t == static_cast<ComputeDataType>(57344)) {
        x = float8_e5m2(x);
      } else {
        x = roundWithTiesToEven(x);
      }
      ComputeDataType max_bound = qmax_t;
      ComputeDataType min_bound = -qmax_t - static_cast<ComputeDataType>(1);
      if (qmax_t == static_cast<ComputeDataType>(448) ||
          qmax_t == static_cast<ComputeDataType>(57344)) {
        min_bound = -qmax_t;
      }
      x = x > max_bound ? max_bound : x;
      x = x < min_bound ? min_bound : x;
      out[i] = static_cast<T>(x);
    } else {
      ComputeDataType v = x > s ? s : x;
      v = v < -s ? -s : v;
      v = qmax_t * inv_s * v;
      out[i] = static_cast<T>(round(v));
    }
  }
}

}  // namespace operators
}  // namespace paddle
