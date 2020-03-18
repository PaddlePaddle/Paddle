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

#include "paddle/fluid/operators/math/bert_encoder_functor.h"
#include "paddle/fluid/operators/math/math_cuda_utils.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T, int TPB>
__device__ inline void LayerNorm(const kvp<T> &thread_data, const int ld,
                                 const int offset, const float *bias,
                                 const float *scale, T *output, T eps) {
  using BlockReduce = cub::BlockReduce<kvp<T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T mu;      // mean
  __shared__ T rsigma;  // 1 / std.dev.

  const auto sum_kv = BlockReduce(temp_storage).Reduce(thread_data, cub::Sum());

  if (threadIdx.x == 0) {
    mu = sum_kv.key;
    rsigma = rsqrt(sum_kv.value - mu * mu + eps);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const T val = output[idx];
    const T g(scale[i]);
    const T b(bias[i]);
    output[idx] = g * (val - mu) * rsigma + b;
  }
}

template <typename T, unsigned TPB>
__global__ void EmbEltwiseLayernormKernel(
    int hidden, const int64_t *word_id_d, const int64_t *pos_id_d,
    const int64_t *sent_id_d, const float *scale, const float *bias,
    const float *word_emb, const float *pos_emb, const float *sent_emb,
    T *output, T eps) {
  cub::Sum pair_sum;
  // blockIdx.x: position in the sequence
  // blockIdx.y: batch
  // gridDim.x: Seq
  // gridDim.y: Batch
  __shared__ int64_t word_id;
  __shared__ int64_t pos_id;
  __shared__ int64_t sent_id;

  const T rhidden = T(1.f) / T(hidden);
  const int seq_pos = blockIdx.y + blockIdx.x * gridDim.y;
  if (threadIdx.x == 0) {
    word_id = word_id_d[seq_pos];
    pos_id = pos_id_d[seq_pos];
    sent_id = sent_id_d[seq_pos];
  }
  __syncthreads();

  // load word, pos, sentence embeddings and add them toghether
  const int64_t woffset = word_id * hidden;
  const int64_t poffset = pos_id * hidden;
  const int64_t soffset = sent_id * hidden;
  const int64_t out_offset = seq_pos * hidden;

  kvp<T> thread_data(0, 0);

#pragma unroll
  for (int it = threadIdx.x; it < hidden; it += TPB) {
    const T w(word_emb[woffset + it]);
    const T p(pos_emb[poffset + it]);
    const T s(sent_emb[soffset + it]);
    const T val = w + s + p;

    output[out_offset + it] = val;
    const T rhiddenval = rhidden * val;

    thread_data = pair_sum(thread_data, kvp<T>(rhiddenval, rhiddenval * val));
  }
  LayerNorm<T, TPB>(thread_data, hidden, out_offset, bias, scale, output, eps);
}

template <typename T>
void EmbEltwiseLayerNormFunctor<T>::operator()(
    int batch, int seq_len, int hidden, const int64_t *word_id_d,
    const int64_t *pos_id_d, const int64_t *sent_id_d, const float *scale,
    const float *bias, const float *word_emb, const float *pos_emb,
    const float *sent_emb, T *output, T eps, cudaStream_t stream) {
  const unsigned tpb = 256;
  const dim3 grid(seq_len, batch, 1);
  const dim3 block(tpb, 1, 1);
  EmbEltwiseLayernormKernel<float, tpb><<<grid, block, 0, stream>>>(
      hidden, word_id_d, pos_id_d, sent_id_d, scale, bias, word_emb, pos_emb,
      sent_emb, output, eps);
}

template class EmbEltwiseLayerNormFunctor<float>;

#ifdef SUPPORT_CUDA_FP16
template class EmbEltwiseLayerNormFunctor<half>;
#endif

}  // namespace math
}  // namespace operators
}  // namespace paddle
