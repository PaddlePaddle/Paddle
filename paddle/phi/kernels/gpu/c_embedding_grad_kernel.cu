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

#include "glog/logging.h"

#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/embedding_grad.h"

COMMON_DECLARE_int64(embedding_deterministic);

namespace phi {

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <typename T, typename IndexT>
__global__ void CEmbeddingGrad(T* table,
                               const T* output,
                               const IndexT* ids,
                               const int rows,
                               const int columns,
                               const int64_t N,
                               const int64_t start_idx,
                               const int64_t end_idx,
                               const int64_t limit) {
  CUDA_KERNEL_LOOP(i, limit) {
    size_t row = i / columns;
    size_t col = i % columns;
    auto id = ids[row];
    if (id >= start_idx && id < end_idx) {
      auto real_idx = id - start_idx;
      phi::CudaAtomicAdd(&table[real_idx * columns + col], output[i]);
    }
  }
}

template <typename T, typename Context>
void CEmbeddingGradKernel(const Context& dev_ctx,
                          const DenseTensor& w,
                          const DenseTensor& ids,
                          const DenseTensor& out_grad,
                          int64_t start_index,
                          DenseTensor* w_grad) {
  int N = w_grad->dims()[0];
  int D = w_grad->dims()[1];
  int K = ids.numel();

  auto limit = K * D;
  int blocks = NumBlocks(limit);
  int threads = kNumCUDAThreads;

  const T* d_output = out_grad.data<T>();
  T* d_table = dev_ctx.template Alloc<T>(w_grad);

  auto t = EigenVector<T>::Flatten(*w_grad);
  t.device(*dev_ctx.eigen_device()) = t.constant(static_cast<T>(0));

  const auto& index_type = ids.dtype();
  if (FLAGS_embedding_deterministic == 1) {
    if (index_type == phi::DataType::INT32) {
      phi::funcs::LaunchEmbeddingGradDeterministicKernel<T, int32_t>(
          dev_ctx,
          ids.data<int32_t>(),
          d_output,
          d_table,
          N,
          D,
          K,
          start_index);
      return;
    } else if (index_type == phi::DataType::INT64) {
      phi::funcs::LaunchEmbeddingGradDeterministicKernel<T, int64_t>(
          dev_ctx,
          ids.data<int64_t>(),
          d_output,
          d_table,
          N,
          D,
          K,
          start_index);
      return;
    }
  } else {
    if (FLAGS_embedding_deterministic > 1) {
      VLOG(2) << "Run grad kernel of embedding with single thread.";
      blocks = 1;
    }
    const int64_t end_idx = start_index + N;
    if (index_type == phi::DataType::INT32) {
      CEmbeddingGrad<T, int32_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(d_table,
                                                     d_output,
                                                     ids.data<int32_t>(),
                                                     K,
                                                     D,
                                                     N,
                                                     start_index,
                                                     end_idx,
                                                     limit);
      return;
    } else if (index_type == phi::DataType::INT64) {
      CEmbeddingGrad<T, int64_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(d_table,
                                                     d_output,
                                                     ids.data<int64_t>(),
                                                     K,
                                                     D,
                                                     N,
                                                     start_index,
                                                     end_idx,
                                                     limit);
      return;
    }
  }
  PADDLE_THROW(common::errors::InvalidArgument(
      "The data type of Input(Ids) must be int32 or int64."));
}

}  // namespace phi

#if (NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000) || \
    defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(c_embedding_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::CEmbeddingGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#else
PD_REGISTER_KERNEL(c_embedding_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::CEmbeddingGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif
