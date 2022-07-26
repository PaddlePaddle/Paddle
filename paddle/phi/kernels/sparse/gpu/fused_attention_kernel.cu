/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/sparse/fused_attention_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/funcs/sparse/sparse_blas.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/matmul_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

namespace phi {
namespace sparse {

#define PRIVATE_CASE_VISIT_ATTN_SOFTMAX(NAME, size, HINT, ...) \
  case size: {                                                 \
    constexpr int HINT = size;                                 \
    __VA_ARGS__();                                             \
    break;                                                     \
  }

#define VISIT_ATTN_SFOTMAX(SIZE, NAME, ...)                                 \
  [&] {                                                                     \
    const auto& __size__ = SIZE;                                            \
    switch (__size__) {                                                     \
      PRIVATE_CASE_VISIT_ATTN_SOFTMAX(NAME, 1, KBufferSize, __VA_ARGS__)    \
      PRIVATE_CASE_VISIT_ATTN_SOFTMAX(NAME, 2, KBufferSize, __VA_ARGS__)    \
      PRIVATE_CASE_VISIT_ATTN_SOFTMAX(NAME, 3, KBufferSize, __VA_ARGS__)    \
      PRIVATE_CASE_VISIT_ATTN_SOFTMAX(NAME, 4, KBufferSize, __VA_ARGS__)    \
      PRIVATE_CASE_VISIT_ATTN_SOFTMAX(NAME, 8, KBufferSize, __VA_ARGS__)    \
      PRIVATE_CASE_VISIT_ATTN_SOFTMAX(NAME, 12, KBufferSize, __VA_ARGS__)   \
      PRIVATE_CASE_VISIT_ATTN_SOFTMAX(NAME, 16, KBufferSize, __VA_ARGS__)   \
      default:                                                              \
        PD_THROW("function " #NAME " is not implemented for columns>512 "); \
    }                                                                       \
  }()

template <typename T, int BufferSize>
__global__ void AttnSoftmaxGpuKernel(const int64_t* x_crows,
                                     const int64_t* x_cols,
                                     const T* x_values,
                                     const T* kp_mask,
                                     const T* attn_mask,
                                     T* out_values,
                                     int M,
                                     int total_row_num,
                                     float scale,
                                     int num_heads,
                                     int batch_nnz) {
  // out = exp(x-x_max) / sum(exp(x-x_max))
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= total_row_num) return;

  int cur_batch = row / M;
  int cur_row = row % M;
  int crow_idx = cur_batch * (M + 1) + cur_row;
  int row_first = cur_batch * batch_nnz + static_cast<int>(x_crows[crow_idx]);
  int row_nnz = static_cast<int>(x_crows[crow_idx + 1] - x_crows[crow_idx]);
  if (row_nnz == 0) return;

  T buffer[BufferSize] = {0};
  int kIteration = (row_nnz + WARP_SIZE - 1) / WARP_SIZE;

  T max_val = -std::numeric_limits<T>::infinity();
  for (int i = 0; i < kIteration; ++i) {
    bool mask = false;
    int idx = threadIdx.x + i * WARP_SIZE;
    if (idx >= row_nnz) break;

    int col_idx = static_cast<int>(x_cols[row_first + idx]);

    if (kp_mask != nullptr &&
        kp_mask[(cur_batch / num_heads) * M + col_idx] == 0) {
      mask = true;
    }
    if (attn_mask != nullptr && attn_mask[cur_row * M + col_idx] == 0) {
      mask = true;
    }

    if (!mask) {
      buffer[i] = x_values[row_first + idx] / scale;
      if (buffer[i] > max_val) {
        max_val = buffer[i];
      }
    }
  }
  T row_max_val = phi::funcs::warpReduceMax<T>(max_val, 0xFFFFFFFF);

  auto functor = phi::funcs::CudaExpFunctor<T>();
  T exp_sum = 0;
  for (int i = 0; i < kIteration; ++i) {
    int idx = threadIdx.x + i * WARP_SIZE;
    if (idx >= row_nnz) break;

    if (buffer[i]) {
      T exp = functor(buffer[i] - row_max_val);
      exp_sum += exp;
      buffer[i] = exp;
    }
  }
  T row_exp_sum = phi::funcs::warpReduceSum<T>(exp_sum, 0xFFFFFFFF);

  for (int i = 0; i < kIteration; ++i) {
    int idx = threadIdx.x + i * WARP_SIZE;
    if (idx >= row_nnz) break;

    if (buffer[i]) {
      out_values[row_first + idx] = buffer[i] / row_exp_sum;
    } else {
      out_values[row_first + idx] = static_cast<T>(0);
    }
  }
}

template <typename T, typename Context>
void FusedAttentionCsrKernel(
    const Context& dev_ctx,
    const DenseTensor& query,
    const DenseTensor& key,
    const DenseTensor& value,
    const SparseCsrTensor& sparse_mask,
    const paddle::optional<DenseTensor>& key_padding_mask,
    const paddle::optional<DenseTensor>& attn_mask,
    DenseTensor* out,
    SparseCsrTensor* softmax) {
#if CUDA_VERSION >= 11070
  /* Check Shape */
  auto q_dim = query.dims();
  auto q_rank = q_dim.size();

  int total_row_num = 1;
  int batch_num = 1;
  for (int i = 0; i < q_rank - 1; ++i) {
    total_row_num *= q_dim[i];
    if (i < q_rank - 2) {
      batch_num *= q_dim[i];
    }
  }
  int M = q_dim[q_rank - 2];
  int N = q_dim[q_rank - 1];

  PADDLE_ENFORCE_EQ(query.dims().size(),
                    4,
                    phi::errors::InvalidArgument(" 'query' must be 4D Tensor"));
  PADDLE_ENFORCE_EQ(key.dims().size(),
                    4,
                    phi::errors::InvalidArgument(" 'key' must be 4D Tensor"));
  PADDLE_ENFORCE_EQ(value.dims().size(),
                    4,
                    phi::errors::InvalidArgument(" 'value' must be 4D Tensor"));

  PADDLE_ENFORCE_EQ(
      sparse_mask.dims().size(),
      3,
      phi::errors::InvalidArgument("dense shape of 'sparse_mask' must be "
                                   "[batch_size*num_heads, seq_len, seq_len]"));
  PADDLE_ENFORCE_EQ(
      sparse_mask.dims()[0],
      q_dim[0] * q_dim[1],
      phi::errors::InvalidArgument("dense shape of 'sparse_mask' must be "
                                   "[batch_size*num_heads, seq_len, seq_len]"));
  PADDLE_ENFORCE_EQ(
      sparse_mask.dims()[1],
      M,
      phi::errors::InvalidArgument("dense shape of 'sparse_mask' must be "
                                   "[batch_size*num_heads, seq_len, seq_len]"));
  PADDLE_ENFORCE_EQ(
      sparse_mask.dims()[2],
      M,
      phi::errors::InvalidArgument("dense shape of 'sparse_mask' must be "
                                   "[batch_size*num_heads, seq_len, seq_len]"));

  const auto kp_mask_ptr = key_padding_mask.get_ptr();
  if (kp_mask_ptr) {
    PADDLE_ENFORCE_EQ(
        kp_mask_ptr->dims().size(),
        2,
        phi::errors::InvalidArgument(
            "shape of 'key_padding_mask' must be [batch_size, seq_len]"));
    PADDLE_ENFORCE_EQ(
        kp_mask_ptr->dims()[0],
        q_dim[0],
        phi::errors::InvalidArgument(
            "shape of 'key_padding_mask' must be [batch_size, seq_len]"));
    PADDLE_ENFORCE_EQ(
        kp_mask_ptr->dims()[1],
        M,
        phi::errors::InvalidArgument(
            "shape of 'key_padding_mask' must be [batch_size, seq_len]"));
  }

  const auto attn_mask_ptr = attn_mask.get_ptr();
  if (attn_mask_ptr) {
    PADDLE_ENFORCE_EQ(attn_mask_ptr->dims().size(),
                      2,
                      phi::errors::InvalidArgument(
                          "shape of 'attn_mask' must be [seq_len, seq_len]"));
    PADDLE_ENFORCE_EQ(attn_mask_ptr->dims()[0],
                      M,
                      phi::errors::InvalidArgument(
                          "shape of 'attn_mask' must be [seq_len, seq_len]"));
    PADDLE_ENFORCE_EQ(attn_mask_ptr->dims()[1],
                      M,
                      phi::errors::InvalidArgument(
                          "shape of 'attn_mask' must be [seq_len, seq_len]"));
  }

  /* Step1: SDD Matmul, reuse */
  SparseCsrTensor sdd_result;
  EmptyLikeCsrKernel<T, Context>(dev_ctx, sparse_mask, &sdd_result);
  auto sparse_blas = phi::funcs::sparse::GetSparseBlas<Context, T>(dev_ctx);
  sparse_blas.SDDMM(false,
                    true,
                    static_cast<T>(1),
                    query,
                    key,
                    static_cast<T>(0),
                    &sdd_result);

  /* Step2: Softmax with kp_mask/attn_mask, manualy not reuse */
  EmptyLikeCsrKernel<T, Context>(dev_ctx, sdd_result, softmax);

  int buffer_size;
  if (M < 128) {
    buffer_size = (M + 32 - 1) / 32;
  } else {
    buffer_size = ((M + 128 - 1) / 128) * 4;
  }

  dim3 grid((total_row_num + 3) / 4);
  dim3 block(WARP_SIZE, 4);

  int batch_nnz = sdd_result.nnz() / batch_num;

  VISIT_ATTN_SFOTMAX(buffer_size, "AttnSoftmaxGpuKernel", [&] {
    AttnSoftmaxGpuKernel<T, KBufferSize><<<grid, block, 0, dev_ctx.stream()>>>(
        sdd_result.non_zero_crows().data<int64_t>(),
        sdd_result.non_zero_cols().data<int64_t>(),
        sdd_result.non_zero_elements().data<T>(),
        kp_mask_ptr ? kp_mask_ptr->data<T>() : nullptr,
        attn_mask_ptr ? attn_mask_ptr->data<T>() : nullptr,
        softmax->mutable_non_zero_elements()->data<T>(),
        M,
        total_row_num,
        std::sqrt(N),
        q_dim[1],
        batch_nnz);
  });

  /* Step3: DSD Matmul, reuse */
  softmax->set_dims(phi::make_ddim({q_dim[0], q_dim[1], q_dim[2], q_dim[2]}));
  MatmulCsrDenseKernel<T, Context>(dev_ctx, *softmax, value, out);
#else
  PADDLE_THROW(
      phi::errors::Unimplemented("forward of 'sparse.nn.functional.attention' "
                                 "use 'cusparseCsrSetStridedBatch', which is "
                                 "completed supported from CUDA 11.7"));
#endif
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(fused_attention_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::FusedAttentionCsrKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
