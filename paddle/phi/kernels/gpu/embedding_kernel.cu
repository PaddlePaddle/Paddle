// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/embedding_kernel.h"
#include "paddle/phi/kernels/funcs/embedding_util.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

template <typename T, typename IdT, bool PaddingFlag>
__global__ void EmbeddingFW(T *output,
                            const T *table,
                            const IdT *ids,
                            const int64_t N,
                            const int64_t K,
                            const int64_t D,
                            const int64_t padding_idx) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * gridDim.x;

  while (idy < K) {
    auto id = static_cast<int64_t>(ids[idy]);
    T *out = output + idy * D;
    const T *tab = table + id * D;
    for (int i = idx; i < D; i += blockDim.x) {
      if (PaddingFlag) {
        if (id == padding_idx)
          out[i] = static_cast<T>(0);
        else
          out[i] = tab[i];
      } else {
        out[i] = tab[i];
      }
    }
    idy += blockDim.y * gridDim.x;
  }
}

template <typename T, typename Context>
struct EmbeddingCUDAFunctor {
  EmbeddingCUDAFunctor(const Context &dev_ctx,
                       const DenseTensor &input,
                       const DenseTensor &weight,
                       int64_t padding_idx,
                       DenseTensor *out)
      : dev_ctx_(dev_ctx),
        input_(input),
        weight_(weight),
        out_(out),
        padding_idx_(padding_idx) {}

  template <typename IdT>
  void apply() {
    size_t N = weight_.dims()[0];
    size_t D = weight_.dims()[1];
    size_t K = input_.numel();

    const int gridx = 2 * dev_ctx_.GetSMCount();
    dim3 threads(256, 4);
    dim3 grids(gridx, 1);

    const T *table = weight_.template data<T>();
    const IdT *ids = input_.template data<IdT>();
    auto *output = dev_ctx_.template Alloc<T>(out_);
    auto stream = dev_ctx_.stream();

    if (padding_idx_ == -1) {
      EmbeddingFW<T, IdT, false><<<grids, threads, 0, stream>>>(
          output, table, ids, N, K, D, padding_idx_);
    } else {
      EmbeddingFW<T, IdT, true><<<grids, threads, 0, stream>>>(
          output, table, ids, N, K, D, padding_idx_);
    }
  }

 private:
  const phi::GPUContext &dev_ctx_;
  const DenseTensor &input_;
  const DenseTensor &weight_;
  DenseTensor *out_;
  int64_t padding_idx_;
};

template <typename T, typename Context>
void EmbeddingKernel(const Context &ctx,
                     const DenseTensor &input,
                     const DenseTensor &weight,
                     int64_t padding_idx,
                     DenseTensor *out) {
  EmbeddingCUDAFunctor<T, Context> functor(
      ctx, input, weight, padding_idx, out);

  if (input.dtype() == phi::DataType::INT32) {
    functor.template apply<int32_t>();
  } else if (input.dtype() == phi::DataType::INT64) {
    functor.template apply<int64_t>();
  } else if (input.dtype() == phi::DataType::INT16) {
    functor.template apply<int16_t>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "emebdding input only support int16, int32 and int64"));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(embedding,
                   GPU,
                   ALL_LAYOUT,
                   phi::EmbeddingKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
