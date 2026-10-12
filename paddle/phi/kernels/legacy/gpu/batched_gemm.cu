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

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <cassert>
#include <cstdint>

#include "paddle/phi/backends/dynload/cublas.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/memory_utils.h"

#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/legacy/gpu/batched_gemm.h"

namespace phi {
namespace {
#define CUBLAS_CALL(func)                                                 \
  do {                                                                    \
    cublasStatus_t status = func;                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                \
      PADDLE_THROW(common::errors::External("cuBLAS error: %d", status)); \
    }                                                                     \
  } while (0)

// datatype mapping
inline cudaDataType_t GetCudaDataType(DataType dtype) {
  switch (dtype) {
    case DataType::FLOAT32:
      return CUDA_R_32F;
    case DataType::FLOAT16:
      return CUDA_R_16F;
    case DataType::BFLOAT16:
      return CUDA_R_16BF;
    default:
      PD_CHECK(false, "Unsupported data type");
      return CUDA_R_32F;
  }
}

// compute type mapping
inline cublasComputeType_t GetCublasComputeType(DataType dtype) {
  switch (dtype) {
    case DataType::FLOAT32:
      return CUBLAS_COMPUTE_32F;
    case DataType::FLOAT16:
      return CUBLAS_COMPUTE_16F;
    case DataType::BFLOAT16:
      return CUBLAS_COMPUTE_32F_FAST_16BF;
    default:
      PD_CHECK(false, "Unsupported data type");
      return CUBLAS_COMPUTE_32F;
  }
}
}  // namespace

template <typename T>
void CublasGemm(cublasHandle_t cublas_handle,
                T *a,
                int64_t a_rows,
                int64_t a_cols,
                bool trans_a,
                T *b,
                int64_t b_rows,
                int64_t b_cols,
                bool trans_b,
                T *c,
                int64_t c_rows,
                int64_t c_cols) {
  // NOTE(Pan Zhaowu): We use int32_t because cuBLAS requires int32_t for
  // arguments. and in most case the matrix's 1D size will not be larger than
  // 2^31.
  const int m = trans_b ? b_rows : b_cols;
  const int k = trans_b ? b_cols : b_rows;
  const int n = trans_a ? a_cols : a_rows;

  const int lda = trans_a ? n : k;
  const int ldb = trans_b ? k : m;
  cublasOperation_t transpose_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transpose_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

  constexpr float alpha = 1.0f, beta = 0.0f;

  if constexpr (std::is_same<T, phi::bfloat16>::value) {
    CUBLAS_CALL(phi::dynload::cublasGemmEx(cublas_handle,
                                           transpose_b,
                                           transpose_a,
                                           m,
                                           n,
                                           k,
                                           &alpha,
                                           b,
                                           CUDA_R_16BF,
                                           ldb,
                                           a,
                                           CUDA_R_16BF,
                                           lda,
                                           &beta,
                                           c,
                                           CUDA_R_16BF,
                                           c_cols,
                                           CUDA_R_32F,
                                           CUBLAS_GEMM_DEFAULT));
  } else if constexpr (std::is_same<T, float>::value) {
    CUBLAS_CALL(phi::dynload::cublasSgemm(cublas_handle,
                                          transpose_b,
                                          transpose_a,
                                          m,
                                          n,
                                          k,
                                          &alpha,
                                          b,
                                          ldb,
                                          a,
                                          lda,
                                          &beta,
                                          c,
                                          c_cols));
  } else {
    PD_CHECK(false, "Unsupported data type in CublasGemm");
  }
}

// Column-output (N == 1) GEMV path, aligned with PyTorch.
//
// Computes out[rows] = mat[rows, cols] (row-major) @ vec[cols].
//
// Two things differ from the generic CublasGemm swap and must be reproduced to
// stay bit-identical with torch.matmul (the accuracy suite's reference):
//   1. Orientation: cuBLAS is column-major, so a row-major GEMM is normally
//      issued with the operands swapped. When the row-major output has a single
//      column torch keeps the un-swapped form (cublasAdjustLdLevel3's
//      `if (n <= 1) ldc = max(m, 1)`), selecting a different kernel / sum
//      order. We mirror that: GEMM(OP_N, OP_N, 1, rows, cols, vec(ld=1),
//      mat(ld=cols), out(ld=1)).
//   2. Output alignment: torch computes each matmul into a fresh (aligned)
//      buffer, while the grouped loop below writes into misaligned offset
//      slices of the big output; cuBLAS then falls back to a different kernel.
//      Callers therefore run this into an aligned scratch buffer and copy the
//      result into place.
template <typename T>
void CublasGemvColumn(cublasHandle_t cublas_handle,
                      const T *mat,
                      const T *vec,
                      T *out,
                      int64_t rows,
                      int64_t cols) {
  constexpr float alpha = 1.0f, beta = 0.0f;
  const int m = 1;
  const int n = static_cast<int>(rows);
  const int k = static_cast<int>(cols);
  if constexpr (std::is_same<T, phi::bfloat16>::value) {
    CUBLAS_CALL(phi::dynload::cublasGemmEx(cublas_handle,
                                           CUBLAS_OP_N,
                                           CUBLAS_OP_N,
                                           m,
                                           n,
                                           k,
                                           &alpha,
                                           vec,
                                           CUDA_R_16BF,
                                           1,
                                           mat,
                                           CUDA_R_16BF,
                                           k,
                                           &beta,
                                           out,
                                           CUDA_R_16BF,
                                           1,
                                           CUDA_R_32F,
                                           CUBLAS_GEMM_DEFAULT));
  } else if constexpr (std::is_same<T, float>::value) {
    CUBLAS_CALL(phi::dynload::cublasSgemm(cublas_handle,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          m,
                                          n,
                                          k,
                                          &alpha,
                                          vec,
                                          1,
                                          mat,
                                          k,
                                          &beta,
                                          out,
                                          1));
  } else {
    PD_CHECK(false, "Unsupported data type in CublasGemvColumn");
  }
}

// Grouped GEMM forward kernel
template <typename T, typename Context>
void m_grouped_gemm_cuda_forward(const Context &dev_ctx,
                                 const DenseTensor &a,
                                 const DenseTensor &b,
                                 const std::vector<int64_t> &batch_sizes,
                                 const bool trans_rhs,
                                 DenseTensor *output) {
  // Get handle according to input tensor, custom_op specific.
  cublasHandle_t handle = dev_ctx.cublas_handle();

  const auto &a_shape = a.dims();
  const auto &b_shape = b.dims();

  const int64_t num_experts = batch_sizes.size();
  const int64_t total_tokens = a_shape[0];
  const int64_t input_hidden_size = a_shape[1];
  const int64_t output_hidden_size = trans_rhs ? b_shape[1] : b_shape[2];

  if constexpr (std::is_same<T, paddle::bfloat16>::value ||
                std::is_same<T, float>::value) {
    T *a_data = const_cast<T *>(a.data<T>());  // alias for a.data
    T *b_data = const_cast<T *>(b.data<T>());  // alias for b.data
    T *output_data = output->data<T>();

    // N == 1 (single output column) is a grouped matrix-vector product. cuBLAS
    // selects a different kernel for this shape depending on operand
    // orientation and output-pointer alignment, so a plain CublasGemm into the
    // misaligned offset slices diverges from torch.matmul by ~1 ulp. Route each
    // expert through the PyTorch-aligned GEMV orientation into an aligned
    // scratch buffer, then copy the column into place.
    if (output_hidden_size == 1) {
      int64_t max_bs = 0;
      for (int64_t i = 0; i < num_experts; ++i) {
        if (batch_sizes[i] > max_bs) max_bs = batch_sizes[i];
      }
      DenseTensor scratch;
      scratch.Resize(make_ddim({max_bs > 0 ? max_bs : 1}));
      T *scr = dev_ctx.template Alloc<T>(&scratch);
      for (int64_t i = 0; i < num_experts; ++i) {
        const int64_t expert_bs = batch_sizes[i];
        if (expert_bs > 0) {
          CublasGemvColumn<T>(
              handle, a_data, b_data, scr, expert_bs, input_hidden_size);
          cudaMemcpyAsync(output_data,
                          scr,
                          expert_bs * sizeof(T),
                          cudaMemcpyDeviceToDevice,
                          dev_ctx.stream());
        }
        a_data += expert_bs * input_hidden_size;
        b_data += b_shape[1] * b_shape[2];
        output_data += expert_bs * output_hidden_size;
      }
      return;
    }

#pragma unroll
    for (int64_t i = 0; i < num_experts; ++i) {
      const int64_t expert_bs = batch_sizes[i];
      CublasGemm(handle,
                 a_data,
                 expert_bs,
                 input_hidden_size,
                 false,
                 b_data,
                 b_shape[1],
                 b_shape[2],
                 trans_rhs,
                 output_data,
                 expert_bs,
                 output_hidden_size);
      a_data += expert_bs * input_hidden_size;
      b_data += b_shape[1] * b_shape[2];
      output_data += expert_bs * output_hidden_size;
    }
  } else {
    PD_CHECK(false, "Unsupported data type, only support bfloat16 and float32");
  }
}

template <typename T, typename Context>
void k_grouped_gemm_cuda_forward(const Context &dev_ctx,
                                 const DenseTensor &a,
                                 const DenseTensor &b,
                                 const std::vector<int64_t> &batch_sizes,
                                 DenseTensor *output) {
  // Get handle according to input tensor, custom_op specific.
  cublasHandle_t handle = dev_ctx.cublas_handle();

  const auto &a_shape = a.dims();
  const auto &b_shape = b.dims();

  const int64_t num_experts = batch_sizes.size();
  const int64_t total_tokens = a_shape[0];
  const int64_t input_hidden_size = a_shape[1];
  const int64_t output_hidden_size = b_shape[1];

  if constexpr (std::is_same<T, paddle::bfloat16>::value ||
                std::is_same<T, float>::value) {
    T *a_data = const_cast<T *>(a.data<T>());  // alias for a.data
    T *b_data = const_cast<T *>(b.data<T>());  // alias for b.data
    T *output_data = output->data<T>();

#pragma unroll
    for (int64_t i = 0; i < num_experts; ++i) {
      const int64_t expert_bs = batch_sizes[i];
      CublasGemm(handle,
                 a_data,
                 expert_bs,
                 input_hidden_size,
                 true,  // trans_lhs
                 b_data,
                 expert_bs,
                 output_hidden_size,
                 false,  // trans_rhs
                 output_data,
                 input_hidden_size,
                 output_hidden_size);
      a_data += expert_bs * input_hidden_size;
      b_data += expert_bs * output_hidden_size;
      output_data += input_hidden_size * output_hidden_size;
    }
  } else {
    PD_CHECK(false, "Unsupported data type, only support bfloat16 and float32");
  }
}

// Expected input:
// a: [total_tokens, input_hidden_size]
// b: [num_experts, input_hidden_size, output_hidden_size] or
//    [num_experts, output_hidden_size, input_hidden_size] if trans_b is true
template <typename T, typename Context>
void BatchedGEMM(const Context &dev_ctx,
                 const DenseTensor &lhs,
                 const DenseTensor &rhs,
                 const std::vector<int64_t> &batch_sizes,
                 const bool trans_lhs,
                 const bool trans_rhs,
                 DenseTensor *output) {
  // see Paddle/paddle/phi/infermeta/binary.cc:BatchedGemmInferMeta for shape
  // infer logics.
  dev_ctx.template Alloc<T>(output);
  // handles 0-size
  if (output->numel() == 0) {
    return;
  }

  const auto lhs_shape = lhs.dims();
  const auto rhs_shape = rhs.dims();
  const int64_t total_tokens = lhs_shape[0];
  const int64_t num_experts = batch_sizes.size();
  // Empty contraction dim: each grouped GEMM contracts over 0 elements, so the
  // mathematically correct result is all zeros. cuBLAS rejects a 0 contraction
  // dim, so fill the output explicitly and skip the GEMM.
  const int64_t contraction = trans_lhs ? total_tokens : lhs_shape[1];
  if (contraction == 0) {
    funcs::SetConstant<Context, T>()(dev_ctx, output, static_cast<T>(0));
    return;
  }
  // We expect layout below:
  // 1. trans_lhs = false && trans_rhs = false (group forward) :
  //    [M_total, input_hidden_size] x [num_experts, input_hidden_size,
  //    output_hidden_size] output: [M_total, output_hidden_size]
  //
  // 2. trans_lhs = false && trans_rhs = true (backward for lhs_grad, or
  // specialized forward):
  //    [M_total, output_hidden_size] x [num_experts, input_hidden_size,
  //    output_hidden_size]' output: [M_total, input_hidden_size]
  //
  // 3. trans_lhs = true && trans_rhs = false (backward for rhs_grad) :
  //    [M_total, input_hidden_size]' x [M_total, output_hidden_size]
  //    output: [num_experts, input_hidden_size, output_hidden_size]

  // TODO(Pan Zhaowu): Using macros to support more dtypes.
  switch (lhs.dtype()) {
    case DataType::BFLOAT16:
      if (!trans_lhs) {
        // =============================================================================
        // Case 1 and 2: group forward or lhs_grad (input_grad)
        // Note that this case implements grouped gemm, mapping hidden_lhs to
        // hidden_out For each expert i, This case views lhs as [Mi x K] and rhs
        // as [E x K x N] or [E x N x K], so the output is [Mtotal x N], N could
        // be input_hidden_size or output_hidden_size.
        m_grouped_gemm_cuda_forward<paddle::bfloat16>(
            dev_ctx, lhs, rhs, batch_sizes, trans_rhs, output);
      } else {
        // =============================================================================
        // Case 3: group backward for rhs_grad (weight_grad)
        // Note that this case implements k-grouped gemm
        // For each expert i, this case views lhs as [K x Mi] and rhs as [Mi x
        // N], so the output is [E x K x N].
        k_grouped_gemm_cuda_forward<paddle::bfloat16>(
            dev_ctx, lhs, rhs, batch_sizes, output);
      }
      break;
    case DataType::FLOAT32:
      if (!trans_lhs) {
        m_grouped_gemm_cuda_forward<float>(
            dev_ctx, lhs, rhs, batch_sizes, trans_rhs, output);
      } else {
        k_grouped_gemm_cuda_forward<float>(
            dev_ctx, lhs, rhs, batch_sizes, output);
      }
      break;
    default:
      PD_CHECK(false, "Unsupported data type");
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(batched_gemm,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchedGEMM,
                   float,
                   double,
                   phi::bfloat16) {}
