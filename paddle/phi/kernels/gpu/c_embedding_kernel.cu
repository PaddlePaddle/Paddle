// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/c_embedding_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <typename T, typename IndexT>
__global__ void CEmbedding(T* out,
                           const T* table,
                           const IndexT* ids,
                           const int rows,
                           const int columns,
                           const int64_t N,
                           const int64_t start_idx,
                           const int64_t end_idx,
                           const int64_t limit,
                           const int64_t vocab_size) {
  CUDA_KERNEL_LOOP(i, limit) {
    size_t row = i / columns;
    size_t col = i % columns;
    auto id = ids[row];

    PADDLE_ENFORCE(
        id >= 0 && (vocab_size < 0 || id < vocab_size),
        "The index is out of bounds, "
        "please check whether the dimensions of index and "
        "input meet the requirements. It should "
        "be less than [%d] and greater than or equal to 0, but received [%d]",
        vocab_size,
        id);
    if (id >= start_idx && id < end_idx) {
      auto real_idx = id - start_idx;
      out[i] = table[real_idx * columns + col];
    } else {
      out[i] = static_cast<T>(0);
    }
  }
}

template <typename T, typename Context>
void CEmbeddingKernel(const Context& ctx,
                      const DenseTensor& w,
                      const DenseTensor& ids,
                      int64_t start_index,
                      int64_t vocab_size,
                      DenseTensor* out) {
  size_t N = w.dims()[0];
  size_t D = w.dims()[1];
  size_t K = ids.numel();

  const int64_t end_idx = start_index + N;

  auto* table = w.data<T>();
  auto* output = ctx.template Alloc<T>(out);

  auto limit = K * D;
  int blocks = NumBlocks(limit);
  int threads = kNumCUDAThreads;

  const auto& index_type = ids.dtype();
  if (index_type == phi::DataType::INT32) {
    CEmbedding<T, int32_t>
        <<<blocks, threads, 0, ctx.stream()>>>(output,
                                               table,
                                               ids.data<int32_t>(),
                                               K,
                                               D,
                                               N,
                                               start_index,
                                               end_idx,
                                               limit,
                                               vocab_size);

  } else if (index_type == phi::DataType::INT64) {
    CEmbedding<T, int64_t>
        <<<blocks, threads, 0, ctx.stream()>>>(output,
                                               table,
                                               ids.data<int64_t>(),
                                               K,
                                               D,
                                               N,
                                               start_index,
                                               end_idx,
                                               limit,
                                               vocab_size);
  } else {
    PADDLE_THROW(common::errors::Unavailable(
        "GPU c_embedding ids only support int32 or int64."));
  }
}
}  // namespace phi

#if (NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000) || \
    defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(c_embedding,
                   GPU,
                   ALL_LAYOUT,
                   phi::CEmbeddingKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#else
PD_REGISTER_KERNEL(c_embedding,
                   GPU,
                   ALL_LAYOUT,
                   phi::CEmbeddingKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif
