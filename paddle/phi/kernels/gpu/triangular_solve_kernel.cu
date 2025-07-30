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

#include "paddle/phi/kernels/triangular_solve_kernel.h"

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

template <typename T, typename Context>
void TriangularSolveKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           bool upper,
                           bool transpose,
                           bool unitriangular,
                           DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  // get broadcast dim
  std::vector<int64_t> x_bst_dims_vec;
  std::vector<int64_t> y_bst_dims_vec;
  std::tie(x_bst_dims_vec, y_bst_dims_vec) =
      funcs::MatrixGetBroadcastDims(x, y);
  int x_bst_ndim = x_bst_dims_vec.size();
  int y_bst_ndim = y_bst_dims_vec.size();

  // Tensor broadcast to 'out' and temp 'x_bst'
  IntArray x_bst_dims(x_bst_dims_vec);
  DenseTensor x_bst = phi::Empty<T, Context>(dev_ctx, x_bst_dims);
  const T* x_bst_data = x_bst.data<T>();
  ExpandKernel<T, Context>(dev_ctx, x, x_bst_dims, &x_bst);

  out->Resize(common::make_ddim(y_bst_dims_vec));
  T* out_data = dev_ctx.template Alloc<T>(out);
  IntArray y_bst_dims(y_bst_dims_vec);
  ExpandKernel<T, Context>(dev_ctx, y, y_bst_dims, out);

  // calculate use cublas library
  CBLAS_UPLO uplo = upper ? CblasUpper : CblasLower;
  CBLAS_TRANSPOSE transA = transpose ? CblasTrans : CblasNoTrans;
  CBLAS_DIAG diag = unitriangular ? CblasUnit : CblasNonUnit;

  int M = static_cast<int>(y_bst_dims_vec[y_bst_ndim - 2]);
  int N = static_cast<int>(y_bst_dims_vec[y_bst_ndim - 1]);
  int lda = std::max(1, M);
  int ldb = std::max(1, N);

  int64_t batch_size = 1;
  for (int64_t i = 0; i < x_bst_ndim - 2; i++) {
    batch_size *= x_bst_dims_vec[i];
  }

  auto blas = phi::funcs::GetBlas<GPUContext, T>(dev_ctx);
  if (batch_size <= 8 && M >= 64) {
    for (int64_t i = 0; i < batch_size; i++) {
      blas.TRSM(CblasLeft,
                uplo,
                transA,
                diag,
                M,
                N,
                T(1),
                x_bst_data + i * M * M,
                lda,
                out_data + i * N * M,
                ldb);
    }
  } else {
    bool use_chunking_workaround = false;
    // Workaround the following a bug on CUDA < 12.1
    // RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling
    // `cublasStrsmBatched
#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)) && \
    defined(CUSOLVER_VERSION) && (CUSOLVER_VERSION < 12100)

    if (N > 524280) {
      use_chunking_workaround = true;
    }
#endif

    if (use_chunking_workaround) {
      constexpr int64_t max_n_size = 524280;
      int64_t n_chunks = (N + max_n_size - 1) / max_n_size;

      std::vector<const T*> cpu_a_ptrs(batch_size);
      for (int64_t i = 0; i < batch_size; ++i) {
        cpu_a_ptrs[i] = x_bst_data + i * M * M;
      }
      phi::Allocator::AllocationPtr gpu_a_ptrs_data = phi::memory_utils::Alloc(
          dev_ctx.GetPlace(),
          cpu_a_ptrs.size() * sizeof(T*),
          phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
      memory_utils::Copy(dev_ctx.GetPlace(),
                         gpu_a_ptrs_data->ptr(),
                         phi::CPUPlace(),
                         static_cast<void*>(cpu_a_ptrs.data()),
                         cpu_a_ptrs.size() * sizeof(T*),
                         dev_ctx.stream());
      const T** gpu_a_ptrs =
          reinterpret_cast<const T**>(gpu_a_ptrs_data->ptr());

      phi::Allocator::AllocationPtr gpu_b_ptrs_data = phi::memory_utils::Alloc(
          dev_ctx.GetPlace(),
          batch_size * sizeof(T*),
          phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
      T** gpu_b_ptrs = reinterpret_cast<T**>(gpu_b_ptrs_data->ptr());

      for (int64_t i = 0; i < n_chunks; ++i) {
        int64_t n_offset = i * max_n_size;
        int current_n =
            static_cast<int>(std::min((int64_t)N - n_offset, max_n_size));

        std::vector<T*> cpu_b_ptrs_for_chunk(batch_size);
        for (int64_t j = 0; j < batch_size; ++j) {
          cpu_b_ptrs_for_chunk[j] = out_data + j * M * N + n_offset;
        }
        memory_utils::Copy(dev_ctx.GetPlace(),
                           gpu_b_ptrs_data->ptr(),
                           phi::CPUPlace(),
                           static_cast<void*>(cpu_b_ptrs_for_chunk.data()),
                           cpu_b_ptrs_for_chunk.size() * sizeof(T*),
                           dev_ctx.stream());

        blas.BatchedTRSM(CblasLeft,
                         uplo,
                         transA,
                         diag,
                         M,
                         current_n,
                         static_cast<T>(1.0),
                         gpu_a_ptrs,
                         lda,
                         gpu_b_ptrs,
                         ldb,
                         batch_size);
      }
    } else {
      std::vector<const T*> cpu_ptrs(batch_size * 2);
      for (int64_t i = 0; i < batch_size; ++i) {
        cpu_ptrs[i] = x_bst_data + i * M * M;
        cpu_ptrs[i + batch_size] = out_data + i * M * N;
      }

      phi::Allocator::AllocationPtr tmp_gpu_ptrs_data =
          phi::memory_utils::Alloc(
              dev_ctx.GetPlace(),
              cpu_ptrs.size() * sizeof(T*),
              phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

      memory_utils::Copy(dev_ctx.GetPlace(),
                         tmp_gpu_ptrs_data->ptr(),
                         phi::CPUPlace(),
                         static_cast<void*>(cpu_ptrs.data()),
                         cpu_ptrs.size() * sizeof(T*),
                         dev_ctx.stream());

      const T** gpu_a_ptrs =
          reinterpret_cast<const T**>(tmp_gpu_ptrs_data->ptr());
      T** gpu_b_ptrs =
          reinterpret_cast<T**>(tmp_gpu_ptrs_data->ptr()) + batch_size;

      blas.BatchedTRSM(CblasLeft,
                       uplo,
                       transA,
                       diag,
                       M,
                       N,
                       static_cast<T>(1.0),
                       gpu_a_ptrs,
                       lda,
                       gpu_b_ptrs,
                       ldb,
                       batch_size);
    }
  }
}

}  // namespace phi

#ifdef PADDLE_WITH_CUDA
PD_REGISTER_KERNEL(triangular_solve,
                   GPU,
                   ALL_LAYOUT,
                   phi::TriangularSolveKernel,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#else  // PADDLE_WITH_HIP
// blas_impl.hip.h not support CUBlas<T>::TRSM for complex
PD_REGISTER_KERNEL(triangular_solve,
                   GPU,
                   ALL_LAYOUT,
                   phi::TriangularSolveKernel,
                   float,
                   double) {}
#endif
