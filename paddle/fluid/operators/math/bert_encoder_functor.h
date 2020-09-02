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
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>  // NOLINT
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct CUDATypeTraits;

#ifdef SUPPORTS_CUDA_FP16
template <>
struct CUDATypeTraits<half> {
  typedef platform::float16 TYPE;
};
#endif

template <>
struct CUDATypeTraits<float> {
  typedef float TYPE;
};

#ifdef PADDLE_WITH_CUDA
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
  void operator()(int batch, int seq_len, int hidden, const int64_t *ids,
                  const float *scale, const float *bias, const int64_t *embs,
                  T *output, float eps, int input_num, cudaStream_t stream);
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
  void operator()(const platform::CUDADeviceContext &dev_ctx, int batch,
                  int seq_len, int head_num, int head_size, T *qkptr,
                  const T *bias_qk_ptr, T *tptr, T alpha, T beta);
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
  void operator()(const int num, const int hidden, const T *input1,
                  const T *input2, const float *scale, const float *bias,
                  T *output, T eps, cudaStream_t stream);
};
#endif

}  // namespace math
}  // namespace operators
}  // namespace paddle
