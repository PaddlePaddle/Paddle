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

#include <cstddef>
#include <cstdint>

#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename IdT>
__global__ void RenormKernel(const T* table,
                             const IdT* ids,
                             T* output,
                             size_t N,
                             size_t D,
                             size_t K,
                             float max_norm,
                             float norm_type) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= K) return;

  IdT id = ids[idx];

  T* out = output + id * D;
  const T* in = table + id * D;

  T norm = 0;
  for (size_t i = 0; i < D; ++i) {
    if (norm_type == 1) {
      norm += fabs(in[i]);
    } else if (norm_type == 2) {
      norm += in[i] * in[i];
    } else {
      norm += powf(in[i], norm_type);
    }
  }
  if (norm_type == 2) {
    norm = sqrtf(norm);
  } else {
    norm = powf(norm, 1.0 / norm_type);
  }

  for (size_t i = 0; i < D; ++i) {
    out[i] = in[i];
  }
  if (norm > max_norm) {
    T scale = max_norm / (norm + 1e-7);
    for (size_t i = 0; i < D; ++i) {
      out[i] = in[i] * scale;
    }
  } else {
    for (size_t i = 0; i < D; ++i) {
      out[i] = in[i];
    }
  }
}

template <typename T, typename Context>
struct EmbeddingRenormGPUFunctor {
  EmbeddingRenormGPUFunctor(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& weight,
                            float max_norm,
                            float norm_type,
                            DenseTensor* out)
      : dev_ctx_(dev_ctx),
        x_(x),
        weight_(weight),
        out_(out),
        max_norm_(max_norm),
        norm_type_(norm_type) {}
  template <typename IdT>
  void apply() {
    size_t N = weight_.dims()[0];
    size_t D = weight_.dims()[1];
    size_t K = x_.numel();

    const T* table = weight_.template data<T>();
    const IdT* ids =
        x_.template data<IdT>();  // TODO(PuQing): speed up by using unique ids
    auto* output = dev_ctx_.template Alloc<T>(out_);

    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx_, K);
    RenormKernel<T, IdT><<<config.block_per_grid,
                           config.thread_per_block,
                           0,
                           dev_ctx_.stream()>>>(
        table, ids, output, N, D, K, max_norm_, norm_type_);
  }

 private:
  const Context& dev_ctx_;
  const DenseTensor& x_;
  const DenseTensor& weight_;
  DenseTensor* out_;
  float max_norm_;
  float norm_type_;
};

template <typename T, typename Context>
void EmbeddingRenormKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& weight,
                           float max_norm,
                           float norm_type,
                           DenseTensor* out) {
  EmbeddingRenormGPUFunctor<T, Context> functor(
      ctx, x, weight, max_norm, norm_type, out);
  if (x.dtype() == phi::DataType::INT32) {
    functor.template apply<int>();
  } else if (x.dtype() == phi::DataType::INT64) {
    functor.template apply<int64_t>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "emebdding input only support int32 and int64, but get %s", x.dtype()));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(embedding_renorm,
                   GPU,
                   ALL_LAYOUT,
                   phi::EmbeddingRenormKernel,
                   float,
                   double) {}
