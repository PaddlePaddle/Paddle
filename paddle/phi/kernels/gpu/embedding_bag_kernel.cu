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

#include "paddle/phi/kernels/embedding_bag_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
namespace phi {

template <typename T, typename IdT, bool PaddingFlag>
__global__ void EmbeddingBag(T *output,
                             const T *table,
                             const IdT *ids,
                             const T *per_sample_weight,
                             const int64_t N,
                             const int64_t K,
                             const int64_t D,
                             const int64_t S,
                             const int64_t padding_idx,
                             CalMode mode) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * gridDim.x;

  while (idy < K / S) {
    T *out = output + idy * D;
    for (int i = idx; i < D; i += blockDim.x) {
      bool choose_max = false;
      int padding_idx_count = 0;
      T sum = static_cast<T>(0);
      T max_d = static_cast<T>(0);
      for (int j = 0; j < S; j ++) {
        auto id = static_cast<int64_t>(ids[idy * S + j]);
        const T *tab = table + id * D;
        if (PaddingFlag && id == padding_idx) {
          padding_idx_count += 1;
        } else {
          sum += tab[i];
          if (!choose_max || max_d < tab[i]) {
            choose_max = true;
            max_d = tab[i];
          }
        }
      }
      if (mode == CalMode::ksum) {
        out[i] = sum;
      } else if (mode == CalMode::kmean) {
        if (padding_idx_count == S) out[i] = static_cast<T>(0);
        else out[i] = sum / (S - padding_idx_count);
      } else {
        out[i] = max_d;
      }
    }
    idy += gridDim.x * blockDim.y;
  }
}

template <typename T, typename Context>
struct EmbeddingBagCUDAFunctor {
  EmbeddingBagCUDAFunctor(const Context &dev_ctx,
                          const DenseTensor &input,
                          const DenseTensor &weight,
                          const DenseTensor &per_sample_weight,
                          const int64_t padding_idx,
                          const std::string &mode,
                          DenseTensor *out)
      : dev_ctx_(dev_ctx),
        input_(input),
        weight_(weight),
        per_sample_weight_(per_sample_weight),
        padding_idx_(padding_idx),
        mode_(mode),
        out_(out) {}

  template <typename IdT>
  void apply() {
    size_t N = weight_.dims()[0];
    size_t D = weight_.dims()[1];
    size_t K = input_.numel();
    size_t S = input_.dims()[1];

    const T *weight_d = weight_.data<T>();
    const IdT *ids_d = input_.data<IdT>();
    const T *per_sample_weight_d = per_sample_weight_.data<T>();
    printf("Before Alloc\n");
    auto *output_d = dev_ctx_.template Alloc<T>(out_);
    auto stream = dev_ctx_.stream();
    printf("After Alloc\n");

    const int gridx = 2 * dev_ctx_.GetSMCount();
    dim3 blocks(256, 4);
    dim3 grids(gridx, 1);

    CalMode mode_enum = CalMode::ksum;
    if (mode_ == "mean") mode_enum = CalMode::kmean;
    if (mode_ == "max") mode_enum = CalMode::kmax;

    if (padding_idx_ == -1) {
      EmbeddingBag<T, IdT, false><<<grids, blocks, 0, stream>>>(
        output_d, weight_d, ids_d, per_sample_weight_d, N, K, D, S, padding_idx_, mode_enum);
    } else {
      EmbeddingBag<T, IdT, true><<<grids, blocks, 0, stream>>>(
        output_d, weight_d, ids_d, per_sample_weight_d, N, K, D, S, padding_idx_, mode_enum);
    }
  }

 private:
  const phi::GPUContext &dev_ctx_;
  const DenseTensor &input_;
  const DenseTensor &weight_;
  const DenseTensor &per_sample_weight_;
  const std::string& mode_;
  const int64_t padding_idx_;
  DenseTensor *out_;
};

template <typename T, typename Context>
void EmbeddingBagCUDAKernel(const Context &ctx,
                            const DenseTensor &input,
                            const DenseTensor &weight,
                            const DenseTensor &per_sample_weight,
                            int64_t padding_idx,
                            const std::string &mode,
                            DenseTensor *out) {
  EmbeddingBagCUDAFunctor<T, Context> functor(
      ctx, input, weight, per_sample_weight, padding_idx, mode, out);
  if (input.dtype() == phi::DataType::INT32) {
    functor.template apply<int32_t>();
  } else if (input.dtype() == phi::DataType::INT64) {
    functor.template apply<int64_t>();
  } else if (input.dtype() == phi::DataType::INT16) {
    functor.template apply<int16_t>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "embebddingbag input only support int16, int32 and int64"));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(embedding_bag,
                   GPU,
                   ALL_LAYOUT,
                   phi::EmbeddingBagCUDAKernel,
                   float,
                   double) {}
