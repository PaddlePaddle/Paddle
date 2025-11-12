// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/embedding_grad_kernel.h"
#include "paddle/phi/kernels/funcs/embedding_grad.h"

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/mixed_vector.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/embedding_util.h"

COMMON_DECLARE_int64(embedding_deterministic);

namespace phi {

template <typename T, typename IndexT>
__global__ void EmbeddingGradAddTo(T* main_grad_out,
                                   const phi::bfloat16* out_grad,
                                   const IndexT* token_indices,
                                   const int64_t num_tokens,
                                   const int64_t token_length) {
  int idx = threadIdx.x;
  int64_t idy =
      static_cast<int64_t>(blockIdx.x) +
      static_cast<int64_t>(threadIdx.y) * static_cast<int64_t>(gridDim.x);

  while (idy < num_tokens) {
    auto id = static_cast<int64_t>(token_indices[idy]);
    const phi::bfloat16* token_out_grad = out_grad + idy * token_length;
    T* token_main_grad = main_grad_out + id * token_length;
    for (int64_t i = idx; i < token_length; i += blockDim.x) {
      phi::CudaAtomicAdd(&token_main_grad[i],
                         static_cast<T>(token_out_grad[i]));
    }
    idy += blockDim.y * gridDim.x;
  }
}

template <typename T, typename Context>
struct EmbeddingGradAddToCUDAFunctor {
  EmbeddingGradAddToCUDAFunctor(const Context& dev_ctx,
                                const DenseTensor& token_indices,
                                const DenseTensor& main_grad_,
                                const DenseTensor& out_grad,
                                DenseTensor* main_grad_out)
      : dev_ctx_(dev_ctx),
        token_indices_(token_indices),
        main_grad_in_(main_grad_),
        out_grad_(out_grad),
        main_grad_out_(main_grad_out) {}

  template <typename IndexT>
  void apply() {
    // Since paddings are not trainable and fixed in forward, the gradient of
    // paddings makes no sense and we don't deal with it in backward.
    {
      size_t token_length = main_grad_out_->dims()[1];
      size_t num_tokens = token_indices_.numel();

      auto main_grad_out_t = main_grad_out_;
      const auto* token_indices = token_indices_.template data<IndexT>();
      T* main_grad_out = dev_ctx_.template Alloc<T>(main_grad_out_t);
      const phi::bfloat16* out_grad = reinterpret_cast<const phi::bfloat16*>(
          out_grad_.template data<phi::bfloat16>());

      const int gridx = 2 * dev_ctx_.GetSMCount();
      dim3 threads(128, 8);
      dim3 grids(gridx, 1);
      EmbeddingGradAddTo<T, IndexT><<<grids, threads, 0, dev_ctx_.stream()>>>(
          main_grad_out, out_grad, token_indices, num_tokens, token_length);
    }
  }

 private:
  const phi::GPUContext& dev_ctx_;
  const DenseTensor& token_indices_;
  const DenseTensor& main_grad_in_;
  const DenseTensor& out_grad_;
  DenseTensor* main_grad_out_;
};

template <typename T, typename Context>
void EmbeddingGradAddToAddToKernel(const Context& dev_ctx,
                                   const DenseTensor& token_indices,
                                   const DenseTensor& main_grad_,
                                   const DenseTensor& out_grad,
                                   DenseTensor* main_grad_out) {
  PADDLE_ENFORCE_EQ(out_grad.dtype(),
                    phi::DataType::BFLOAT16,
                    "out_grad dtype must be bfloat16 in embedding_grad_add_to");
  EmbeddingGradAddToCUDAFunctor<T, Context> functor(
      dev_ctx, token_indices, main_grad_, out_grad, main_grad_out);

  if (token_indices.dtype() == phi::DataType::INT32) {
    functor.template apply<int>();
  } else if (token_indices.dtype() == phi::DataType::INT64) {
    functor.template apply<int64_t>();
  } else if (token_indices.dtype() == phi::DataType::INT16) {
    functor.template apply<int16_t>();
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "embedding token_indices only support int16, int32 and int64"));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(embedding_grad_add_to,
                   GPU,
                   ALL_LAYOUT,
                   phi::EmbeddingGradAddToAddToKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
