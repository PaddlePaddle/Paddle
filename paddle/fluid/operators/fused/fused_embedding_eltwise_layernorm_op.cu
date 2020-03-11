// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda_runtime.h>
#include <paddle/fluid/platform/device_context.h>
#include <algorithm>
#include <cub/cub.cuh>  // NOLINT
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

template <typename T>
using kvp = cub::KeyValuePair<T, T>;

template <typename T>
using cv2 = cub::CubVector<T, 2>;

template <typename T, int TPB>
__device__ inline void LayerNorm(const cv2<T> &thread_data, const int ld,
                                 const int offset, const float *bias,
                                 const float *scale, T *output, float eps) {
  using BlockReduce = cub::BlockReduce<cv2<T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T mu;      // mean
  __shared__ T rsigma;  // 1 / std.dev.

  const auto sum_kv = BlockReduce(temp_storage).Reduce(thread_data, cub::Sum());

  if (threadIdx.x == 0) {
    mu = sum_kv.x;
    rsigma = rsqrt(sum_kv.y - mu * mu + eps);
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
    const int64_t *sent_id_d, const T *scale, const T *bias, const T *word_emb,
    const T *pos_emb, const T *sent_emb, T *output, float eps) {
  cub::Sum pair_sum;
  // blockIdx.x: position in the sequence
  // blockIdx.y: batch
  // gridDim.x: Seq
  // gridDim.y: Batch
  __shared__ int64_t word_id;
  __shared__ int64_t pos_id;
  __shared__ int64_t sent_id;

  const T rhidden = T(1.f) / T(hidden);
  const int64_t seq_pos = blockIdx.y + blockIdx.x * gridDim.y;
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

  cv2<T> thread_data;
  thread_data.x = 0;
  thread_data.y = 0;

#pragma unroll
  for (int it = threadIdx.x; it < hidden; it += TPB) {
    const T w(word_emb[woffset + it]);
    const T p(pos_emb[poffset + it]);
    const T s(sent_emb[soffset + it]);
    const T val = w + s + p;

    output[out_offset + it] = val;
    const T rhiddenval = rhidden * val;
    cv2<T> temp_data;
    temp_data.x = rhiddenval;
    temp_data.y = rhiddenval * val;

    thread_data = pair_sum(thread_data, temp_data);
  }
  LayerNorm<T, TPB>(thread_data, hidden, out_offset, bias, scale, output, eps);
}

template <typename DeviceContext, typename T>
class EmbeddingEltWiseLayerNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    using Tensor = framework::Tensor;
    auto *word_id = context.Input<framework::Tensor>("WordId");
    auto *pos_id = context.Input<framework::Tensor>("PosId");
    auto *sent_id = context.Input<framework::Tensor>("SentId");

    auto *word_emb = context.Input<framework::Tensor>("WordEmb");
    auto *pos_emb = context.Input<framework::Tensor>("PosEmb");
    auto *sent_emb = context.Input<framework::Tensor>("SentEmb");

    auto *bias = context.Input<framework::Tensor>("Bias");
    auto *scale = context.Input<framework::Tensor>("Scale");
    auto *out = context.Output<framework::Tensor>("Out");

    auto *word_id_d = word_id->data<int64_t>();
    auto *pos_id_d = pos_id->data<int64_t>();
    auto *sent_id_d = sent_id->data<int64_t>();

    auto *word_emb_d = word_emb->data<T>();
    auto *pos_emb_d = pos_emb->data<T>();
    auto *sent_emb_d = sent_emb->data<T>();

    auto *bias_d = bias->data<T>();
    auto *scale_d = scale->data<T>();
    auto *output_d = out->mutable_data<T>(context.GetPlace());
    // compute q*k with eltadd
    auto &device_ctx = context.template device_context<DeviceContext>();
    float eps = context.Attr<float>("epsilon");

    // should be (B * S * hidden)
    auto word_id_dims = word_id->dims();
    auto word_emb_dims = word_emb->dims();

    int batch = word_id_dims[0];
    int seq_len = word_id_dims[1];
    int hidden = word_emb_dims[1];

    const unsigned tpb = 256;
    const dim3 grid(seq_len, batch, 1);
    const dim3 block(tpb, 1, 1);
    EmbEltwiseLayernormKernel<T, tpb><<<grid, block, 0, device_ctx.stream()>>>(
        hidden, word_id_d, pos_id_d, sent_id_d, scale_d, bias_d, word_emb_d,
        pos_emb_d, sent_emb_d, output_d, eps);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_embedding_eltwise_layernorm,
                        ops::EmbeddingEltWiseLayerNormKernel<
                            paddle::platform::CUDADeviceContext, float>);
