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
__global__ void ClipAndQuantKernel(const T *in,
                                   const T *scale,
                                   const int bin_cnt,
                                   const int round_type,
                                   const int n,
                                   T *out) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  using ComputeDataType = typename QuantizeDataType<T>::type;

  ComputeDataType s = static_cast<ComputeDataType>(scale[0]);
  ComputeDataType inv_s = phi::funcs::inverse(s);
  ComputeDataType bin_cnt_t = static_cast<ComputeDataType>(bin_cnt);

  for (int i = bid; i < n; i += blockDim.x * gridDim.x) {
    ComputeDataType x = static_cast<ComputeDataType>(in[i]);
    if (round_type == 0) {
      x = bin_cnt_t * inv_s * x;
      x = phi::funcs::roundWithTiesToEven(x);
      ComputeDataType max_bound = bin_cnt_t;
      ComputeDataType min_bound = -bin_cnt_t - static_cast<ComputeDataType>(1);
      x = x > max_bound ? max_bound : x;
      x = x < min_bound ? min_bound : x;
      out[i] = static_cast<T>(x);
    } else {
      ComputeDataType v = x > s ? s : x;
      v = v < -s ? -s : v;
      v = bin_cnt_t * inv_s * v;
      out[i] = static_cast<T>(round(v));
    }
  }
}

}  // namespace operators
}  // namespace paddle
