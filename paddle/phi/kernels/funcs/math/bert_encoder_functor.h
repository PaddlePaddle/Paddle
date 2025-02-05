// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>  // NOLINT
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>

#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/float16.h"

namespace phi {
namespace math {

template <typename T>
struct CUDATypeTraits;

template <>
struct CUDATypeTraits<half> {
  typedef phi::dtype::float16 TYPE;
};

template <>
struct CUDATypeTraits<float> {
  typedef float TYPE;
};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// This functor involves a fusion calculation in Ernie or Bert.
// The fusion mode is as follows:
//
// |           |
// other_op1   other_op2
//      |           |
//      |------elementwise_add
//                  |
//              layer_norm
//                  |
//              other_op3
//                  |

template <typename T>
class SkipLayerNormFunctor {
 public:
  void operator()(const int num,
                  const int hidden,
                  const T *input1,
                  const T *input2,
                  const T *scale,
                  const T *bias,
                  T *output,
                  float eps,
                  gpuStream_t stream);
};
#endif

}  // namespace math
}  // namespace phi
