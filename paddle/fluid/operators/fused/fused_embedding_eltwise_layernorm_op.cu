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
#include "paddle/fluid/framework/framework.pb.h"
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
__global__ void EmbEltwiseLayernormKernel(int hidden, const int64_t *ids,
                                          const T *scale, const T *bias,
                                          const int64_t *embs, T *output,
                                          float eps, int input_num) {
  cub::Sum pair_sum;
  // blockIdx.x: position in the sequence
  // blockIdx.y: batch
  // gridDim.x: Seq
  // gridDim.y: Batch

  extern __shared__ int64_t array_id[];

  const T rhidden = T(1.f) / T(hidden);
  const int64_t seq_pos = blockIdx.y + blockIdx.x * gridDim.y;
  if (threadIdx.x == 0) {
    for (int i = 0; i < input_num; ++i) {
      const int64_t *ids_p = reinterpret_cast<const int64_t *>(ids[i]);
      array_id[i] = ids_p[seq_pos];
    }
  }
  __syncthreads();

  const int64_t out_offset = seq_pos * hidden;

  cv2<T> thread_data;
  thread_data.x = 0;
  thread_data.y = 0;

#pragma unroll
  for (int it = threadIdx.x; it < hidden; it += TPB) {
    T val = 0;
    for (int i = 0; i < input_num; ++i) {
      val += reinterpret_cast<const T *>(embs[i])[array_id[i] * hidden + it];
    }

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
    auto &device_ctx = context.template device_context<DeviceContext>();
    auto ids = context.MultiInput<framework::Tensor>("Ids");
    auto embs = context.MultiInput<framework::Tensor>("Embs");
    int input_num = static_cast<int>(ids.size());

    framework::Tensor in_ids_(framework::proto::VarType::INT64),
        in_embs_(framework::proto::VarType::INT64);
    framework::DDim in_dim{input_num};
    int device_id;
    cudaGetDevice(&device_id);
    in_ids_.Resize(in_dim);
    in_embs_.Resize(in_dim);
    int64_t *in_ids_d =
        in_ids_.mutable_data<int64_t>(platform::CUDAPlace(device_id));
    int64_t *in_embs_d =
        in_embs_.mutable_data<int64_t>(platform::CUDAPlace(device_id));

    std::vector<int64_t> in1s, in2s;
    for (int i = 0; i < input_num; ++i) {
      in1s.push_back(reinterpret_cast<uintptr_t>(ids[i]->data<int64_t>()));
      in2s.push_back(reinterpret_cast<uintptr_t>(embs[i]->data<T>()));
    }

    cudaMemcpyAsync(in_ids_d, in1s.data(), sizeof(int64_t) * input_num,
                    cudaMemcpyHostToDevice, device_ctx.stream());
    cudaMemcpyAsync(in_embs_d, in2s.data(), sizeof(int64_t) * input_num,
                    cudaMemcpyHostToDevice, device_ctx.stream());

    auto *bias = context.Input<framework::Tensor>("Bias");
    auto *scale = context.Input<framework::Tensor>("Scale");
    auto *out = context.Output<framework::Tensor>("Out");

    // should be (B * S * hidden)
    auto id0_dims = ids[0]->dims();
    auto emb0_dims = embs[0]->dims();

    int batch = id0_dims[0];
    int seq_len = id0_dims[1];
    int hidden = emb0_dims[1];

    auto *bias_d = bias->data<T>();
    auto *scale_d = scale->data<T>();
    auto *output_d = out->mutable_data<T>(context.GetPlace());
    float eps = context.Attr<float>("epsilon");

    const unsigned tpb = 256;
    const dim3 grid(seq_len, batch, 1);
    const dim3 block(tpb, 1, 1);
    int shared_bytes = input_num * sizeof(int64_t);
    EmbEltwiseLayernormKernel<
        T, tpb><<<grid, block, shared_bytes, device_ctx.stream()>>>(
        hidden, in_ids_d, scale_d, bias_d, in_embs_d, output_d, eps, input_num);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_embedding_eltwise_layernorm,
                        ops::EmbeddingEltWiseLayerNormKernel<
                            paddle::platform::CUDADeviceContext, float>);
