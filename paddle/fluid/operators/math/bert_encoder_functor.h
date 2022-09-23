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

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct CUDATypeTraits;

template <>
struct CUDATypeTraits<half> {
  typedef platform::float16 TYPE;
};

template <>
struct CUDATypeTraits<float> {
  typedef float TYPE;
};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// This functor involves a fusion calculation in Ernie or Bert.
//  The fusion mode is as follows:
//
//      in_var  emb       in_var   emb
//        |      |          |       |
//      lookup_table      lookup_table
//            |                 |
//         lkt_var           lkt_var
//             \                /
//              elementwise_add
//                     |
//                elt_out_var
//
template <typename T>
class EmbEltwiseLayerNormFunctor {
 public:
  void operator()(int batch,
                  int seq_len,
                  int hidden,
                  const int64_t *ids,
                  const T *scale,
                  const T *bias,
                  const int64_t *embs,
                  T *output,
                  float eps,
                  int input_num,
                  gpuStream_t stream);
};

// This functor involves a fusion calculation in Ernie or Bert.
// The fusion mode is as follows:
//
//         |    |
//         matmul
//           |
//       eltwise_add
//           |
//        softmax    /
//           \      /
//             matmul
//               |

template <typename T>
class MultiHeadGPUComputeFunctor {
 public:
  void operator()(const phi::GPUContext &dev_ctx,
                  int batch,
                  int seq_len,
                  int head_num,
                  int head_size,
                  T *qkptr,
                  const T *bias_qk_ptr,
                  T *tptr,
                  T alpha,
                  T beta);
};

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
}  // namespace operators
}  // namespace paddle
