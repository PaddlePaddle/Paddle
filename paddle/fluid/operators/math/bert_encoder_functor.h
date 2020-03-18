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

namespace paddle {
namespace operators {
namespace math {

#ifdef PADDLE_WITH_CUDA
template <typename T>
class EmbEltwiseLayerNormFunctor {
 public:
  void operator()(int batch, int seq_len, int hidden, const int64_t *word_id_d,
                  const int64_t *pos_id_d, const int64_t *sent_id_d,
                  const float *scale, const float *bias, const float *word_emb,
                  const float *pos_emb, const float *sent_emb, T *output, T eps,
                  cudaStream_t stream);
};
#endif

}  // namespace math
}  // namespace operators
}  // namespace paddle
