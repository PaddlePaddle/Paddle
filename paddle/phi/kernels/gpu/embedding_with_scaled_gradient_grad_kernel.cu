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

#include "paddle/phi/kernels/embedding_with_scaled_gradient_grad_kernel.h"
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
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/embedding_util.h"

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

using phi::PADDLE_CUDA_NUM_THREADS;
COMMON_DECLARE_int64(embedding_deterministic);

inline int GET_BLOCKS(const int N) {
  return (N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS;
}

namespace phi {

template <typename InT, typename OutT>
__global__ void InputTypeConvert(const InT* in_ids,
                                 const int64_t K,
                                 OutT* out_ids) {
  for (int i = 0; i < K; i++) {
    out_ids[i] = static_cast<OutT>(in_ids[i]);
  }
}

template <typename T, typename IdT>
__global__ void EmbeddingGrad(T* table,
                              const T* output,
                              const IdT* ids,
                              const int64_t N,
                              const int64_t K,
                              const int64_t D) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * gridDim.x;

  while (idy < K) {
    auto id = static_cast<int64_t>(ids[idy]);
    const T* out = output + idy * D;
    T* tab = table + id * D;
#ifdef PADDLE_WITH_CUDA
    phi::VectorizedAtomicAddPerBlock(D, idx, blockDim.x, out, tab);
#else
    for (int i = idx; i < D; i += blockDim.x) {
      phi::CudaAtomicAdd(&tab[i], out[i]);
    }
#endif
    idy += blockDim.y * gridDim.x;
  }
}

template <typename IdT>
__global__ void CountFreqKernel(const IdT* ids_data,
                                int64_t num_ids,
                                int64_t num_weights,
                                int* count_data) {
  extern __shared__ int buf_count[];
  for (int i = threadIdx.x; i < num_weights; i += blockDim.x) {
    buf_count[i] = 0;
  }
  __syncthreads();

  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_ids) {
    phi::CudaAtomicAdd(&buf_count[ids_data[idx]], 1);
  }

  __syncthreads();

  for (int i = threadIdx.x; i < num_weights; i += blockDim.x) {
    phi::CudaAtomicAdd(&count_data[i], buf_count[i]);
  }
}

template <typename T>
__global__ void ScaleGradKernel(const int* count_data,
                                int64_t num_weights,
                                int64_t num_weight_dim,
                                T* table) {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_weights) {
    MPType freq = static_cast<MPType>(count_data[idx]);
    freq = (freq == static_cast<MPType>(0)) ? 1 : freq;
    for (int i = 0; i < num_weight_dim; ++i) {
      MPType scaled_grad =
          static_cast<MPType>(table[idx * num_weight_dim + i]) / freq;
      table[idx * num_weight_dim + i] = static_cast<T>(scaled_grad);
    }
  }
}

template <typename T, typename Context>
struct EmbeddingWithScaledGradientGradCUDAFunctor {
  EmbeddingWithScaledGradientGradCUDAFunctor(const Context& dev_ctx,
                                             const DenseTensor& input,
                                             const DenseTensor& weight,
                                             const DenseTensor& out_grad,
                                             int64_t padding_idx,
                                             DenseTensor* weight_grad)
      : dev_ctx_(dev_ctx),
        input_(input),
        weight_(weight),
        out_grad_(out_grad),
        padding_idx_(padding_idx),
        weight_grad_(weight_grad) {}

  template <typename IdT>
  void apply() {
    // Since paddings are not trainable and fixed in forward, the gradient of
    // paddings makes no sense and we don't deal with it in backward.
    {
      auto d_output_t = out_grad_;
      auto d_table_t = weight_grad_;

      size_t N = weight_grad_->dims()[0];
      size_t D = weight_grad_->dims()[1];
      size_t K = input_.numel();

      const T* d_output = d_output_t.template data<T>();
      const auto* ids = input_.template data<IdT>();
      T* d_table = dev_ctx_.template Alloc<T>(d_table_t);

#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipMemsetAsync(d_table, 0, N * D * sizeof(T), dev_ctx_.stream()));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemsetAsync(d_table, 0, N * D * sizeof(T), dev_ctx_.stream()));
#endif

      if (FLAGS_embedding_deterministic == 1) {
        phi::funcs::LaunchEmbeddingGradDeterministicKernel<T, IdT>(
            dev_ctx_, ids, d_output, d_table, N, D, K);
      } else {
        const int gridx = 2 * dev_ctx_.GetSMCount();
        dim3 threads(128, 8);
        dim3 grids(gridx, 1);
        if (FLAGS_embedding_deterministic > 1) {
          VLOG(2) << "Run grad kernel of embedding with single thread.";
          grids.x = 1;
          threads.y = 1;
        }
        EmbeddingGrad<T, IdT><<<grids, threads, 0, dev_ctx_.stream()>>>(
            d_table, d_output, ids, N, K, D);
      }

      DenseTensor count_ids =
          phi::Empty<int, Context>(dev_ctx_, {static_cast<int64_t>(N)});
      int* count_ids_data = count_ids.data<int>();
      auto stream = dev_ctx_.stream();
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipMemsetAsync(count_ids_data, 0, N * sizeof(int), stream));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemsetAsync(count_ids_data, 0, N * sizeof(int), stream));
#endif
      CountFreqKernel<IdT>
          <<<GET_BLOCKS(K), PADDLE_CUDA_NUM_THREADS, N * sizeof(int), stream>>>(
              ids, K, N, count_ids_data);
      ScaleGradKernel<T><<<GET_BLOCKS(N), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
          count_ids_data, N, D, d_table);
    }
  }

 private:
  const phi::GPUContext& dev_ctx_;
  const DenseTensor& input_;
  const DenseTensor& weight_;
  const DenseTensor& out_grad_;
  int64_t padding_idx_;
  DenseTensor* weight_grad_;
};

template <typename T, typename Context>
void EmbeddingWithScaledGradientGradKernel(const Context& ctx,
                                           const DenseTensor& input,
                                           const DenseTensor& weight,
                                           const DenseTensor& out_grad,
                                           int64_t padding_idx,
                                           DenseTensor* weight_grad) {
  EmbeddingWithScaledGradientGradCUDAFunctor<T, Context> functor(
      ctx, input, weight, out_grad, padding_idx, weight_grad);
  if (input.dtype() == phi::DataType::INT32) {
    functor.template apply<int>();
  } else if (input.dtype() == phi::DataType::INT64) {
    functor.template apply<int64_t>();
  } else if (input.dtype() == phi::DataType::INT16) {
    functor.template apply<int16_t>();
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "embedding input only support int16, int32 and int64"));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(embedding_with_scaled_gradient_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::EmbeddingWithScaledGradientGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
