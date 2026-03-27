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

#ifdef PADDLE_WITH_MAGMA

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/cublas.h"
#include "paddle/phi/backends/dynload/cusolver.h"
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
#include "hip/hip_runtime.h"
#include "paddle/phi/backends/dynload/rocblas.h"
#include "paddle/phi/backends/dynload/rocsolver.h"
#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"
#endif  // PADDLE_WITH_HIP

#endif  // PADDLE_WITH_MAGMA

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/cpu/eig.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/unsqueeze.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

#ifdef PADDLE_WITH_MAGMA
template <typename T>
void SolveLinearSystemGPU(const GPUContext& dev_ctx,
                          const T* matrix_data,
                          const T* rhs_data,
                          T* out_data,
                          int order,
                          int rhs_cols,
                          int batch_count);

#ifdef PADDLE_WITH_CUDA
template <>
void SolveLinearSystemGPU<phi::dtype::complex<float>>(
    const GPUContext& dev_ctx,
    const phi::dtype::complex<float>*
        matrix_data,  // device ptr, row-major, size batch*order*order
    const phi::dtype::complex<float>*
        rhs_data,  // device ptr, row-major, size batch*order*rhs_cols
    phi::dtype::complex<float>*
        out_data,  // device ptr, row-major, size batch*order*rhs_cols
    int order,
    int rhs_cols,
    int batch_count) {
  // handles
  cublasHandle_t cublas_handle = dev_ctx.cublas_handle();
  cusolverDnHandle_t cusolver_handle = dev_ctx.cusolver_dn_handle();
  auto stream = phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream()));

  // cuComplex constants
  const cuComplex kAlpha = make_cuFloatComplex(1.0f, 0.0f);
  const cuComplex kZero = make_cuFloatComplex(0.0f, 0.0f);

  // Sizes
  const size_t A_one_bytes =
      static_cast<size_t>(order) * order * sizeof(cuComplex);
  const size_t B_one_bytes =
      static_cast<size_t>(order) * rhs_cols * sizeof(cuComplex);
  const size_t A_batch_bytes = A_one_bytes * batch_count;
  const size_t B_batch_bytes = B_one_bytes * batch_count;

  const cuComplex* A_row_all = reinterpret_cast<const cuComplex*>(matrix_data);
  const cuComplex* B_row_all = reinterpret_cast<const cuComplex*>(rhs_data);
  cuComplex* X_row_all = reinterpret_cast<cuComplex*>(out_data);

  auto dA_col_alloc =
      phi::memory_utils::Alloc(dev_ctx.GetPlace(), A_batch_bytes, stream);
  auto dB_col_alloc =
      phi::memory_utils::Alloc(dev_ctx.GetPlace(), B_batch_bytes, stream);
  cuComplex* dA_col = reinterpret_cast<cuComplex*>(dA_col_alloc->ptr());
  cuComplex* dB_col = reinterpret_cast<cuComplex*>(dB_col_alloc->ptr());

  auto d_pivots_alloc = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      static_cast<size_t>(batch_count) * order * sizeof(int),
      stream);
  int* d_pivots = reinterpret_cast<int*>(d_pivots_alloc->ptr());

  auto d_info_alloc =
      phi::memory_utils::Alloc(dev_ctx.GetPlace(),
                               static_cast<size_t>(batch_count) * sizeof(int),
                               stream);
  int* d_info = reinterpret_cast<int*>(d_info_alloc->ptr());

  //    A_row layout: row-major (order x order), B_row layout: row-major (order
  //    x rhs_cols)
  for (int i = 0; i < batch_count; ++i) {
    const cuComplex* A_row = A_row_all + static_cast<size_t>(i) * order * order;
    cuComplex* A_col = dA_col + static_cast<size_t>(i) * order * order;
    const cuComplex* B_row =
        B_row_all + static_cast<size_t>(i) * order * rhs_cols;
    cuComplex* B_col = dB_col + static_cast<size_t>(i) * order * rhs_cols;

    // transpose A_row (row-major) -> A_col (column-major) via C = A^T
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasCgeam(cublas_handle,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             order,
                             order,
                             &kAlpha,
                             A_row,
                             order,  // lda: when interpreting A_row as (order x
                                     // order) row-major, using order
                             &kZero,
                             nullptr,
                             order,
                             A_col,
                             order));  // ldc = order (column-major leading dim)

    // transpose B_row (row-major order x rhs_cols) -> B_col (column-major order
    // x rhs_cols)
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasCgeam(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        order,
        rhs_cols,
        &kAlpha,
        B_row,
        rhs_cols,  // lda when A_row is viewed row-major: leading = rhs_cols
        &kZero,
        nullptr,
        rhs_cols,
        B_col,
        order));  // ldc = order
  }

  int lwork = 0;
  cuComplex* dA_col0 = dA_col;
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnCgetrf_bufferSize(
      cusolver_handle, order, order, dA_col0, order, &lwork));

  size_t work_bytes = static_cast<size_t>(lwork) * sizeof(cuComplex);
  auto d_work_alloc =
      phi::memory_utils::Alloc(dev_ctx.GetPlace(), work_bytes, stream);
  cuComplex* d_work = reinterpret_cast<cuComplex*>(d_work_alloc->ptr());

  for (int i = 0; i < batch_count; ++i) {
    cuComplex* A_col = dA_col + static_cast<size_t>(i) * order * order;
    cuComplex* B_col = dB_col + static_cast<size_t>(i) * order * rhs_cols;
    int* pivots_i = d_pivots + static_cast<size_t>(i) * order;
    int* info_i = d_info + i;

    // getrf (LU factorization) on A_col (column-major)
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnCgetrf(
        cusolver_handle, order, order, A_col, order, d_work, pivots_i, info_i));

    // getrs: solve A_col * X_col = B_col
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnCgetrs(
        cusolver_handle,
        CUBLAS_OP_N,  // no transpose on column-major matrix
        order,
        rhs_cols,
        A_col,
        order,
        pivots_i,
        B_col,
        order,
        info_i));
  }

  for (int i = 0; i < batch_count; ++i) {
    cuComplex* B_col = dB_col + static_cast<size_t>(i) * order *
                                    rhs_cols;  // X in column-major
    cuComplex* X_row = X_row_all + static_cast<size_t>(i) * order *
                                       rhs_cols;  // target row-major

    // transpose X_col -> X_row
    // We use C = A^T : A has shape (order x rhs_cols) in column-major, so C
    // will be (rhs_cols x order), but we want X_row with shape (order x
    // rhs_cols) in row-major; calling cublasCgeam with op=T and adjusted dims
    // works:
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasCgeam(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        rhs_cols,
        order,  // rowsC = rhs_cols, colsC = order
        &kAlpha,
        B_col,
        order,  // B_col lda = order (col-major)
        &kZero,
        nullptr,
        order,
        X_row,
        rhs_cols));  // X_row ldc = rhs_cols (row-major leading dimension)
  }

  std::vector<int> h_info(batch_count, 0);
  phi::memory_utils::Copy(CPUPlace(),
                          h_info.data(),
                          dev_ctx.GetPlace(),
                          d_info,
                          static_cast<size_t>(batch_count) * sizeof(int),
                          reinterpret_cast<void*>(dev_ctx.stream()));
  dev_ctx.Wait();

  for (int i = 0; i < batch_count; ++i) {
    PADDLE_ENFORCE_EQ(
        h_info[i],
        0,
        errors::External(
            "cuSOLVER getrf/getrs failed at batch %d, info: %d", i, h_info[i]));
  }
}

template <>
void SolveLinearSystemGPU<phi::dtype::complex<double>>(
    const GPUContext& dev_ctx,
    const phi::dtype::complex<double>*
        matrix_data,  // device ptr, row-major, size batch*order*order
    const phi::dtype::complex<double>*
        rhs_data,  // device ptr, row-major, size batch*order*rhs_cols
    phi::dtype::complex<double>*
        out_data,  // device ptr, row-major, size batch*order*rhs_cols
    int order,
    int rhs_cols,
    int batch_count) {
  // handles
  cublasHandle_t cublas_handle = dev_ctx.cublas_handle();
  cusolverDnHandle_t cusolver_handle = dev_ctx.cusolver_dn_handle();
  auto stream = phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream()));

  // cuDoubleComplex constants
  const cuDoubleComplex kAlpha = make_cuDoubleComplex(1.0f, 0.0f);
  const cuDoubleComplex kZero = make_cuDoubleComplex(0.0f, 0.0f);

  // Sizes
  const size_t A_one_bytes =
      static_cast<size_t>(order) * order * sizeof(cuDoubleComplex);
  const size_t B_one_bytes =
      static_cast<size_t>(order) * rhs_cols * sizeof(cuDoubleComplex);
  const size_t A_batch_bytes = A_one_bytes * batch_count;
  const size_t B_batch_bytes = B_one_bytes * batch_count;

  const cuDoubleComplex* A_row_all =
      reinterpret_cast<const cuDoubleComplex*>(matrix_data);
  const cuDoubleComplex* B_row_all =
      reinterpret_cast<const cuDoubleComplex*>(rhs_data);
  cuDoubleComplex* X_row_all = reinterpret_cast<cuDoubleComplex*>(out_data);

  auto dA_col_alloc =
      phi::memory_utils::Alloc(dev_ctx.GetPlace(), A_batch_bytes, stream);
  auto dB_col_alloc =
      phi::memory_utils::Alloc(dev_ctx.GetPlace(), B_batch_bytes, stream);
  cuDoubleComplex* dA_col =
      reinterpret_cast<cuDoubleComplex*>(dA_col_alloc->ptr());
  cuDoubleComplex* dB_col =
      reinterpret_cast<cuDoubleComplex*>(dB_col_alloc->ptr());

  auto d_pivots_alloc = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      static_cast<size_t>(batch_count) * order * sizeof(int),
      stream);
  int* d_pivots = reinterpret_cast<int*>(d_pivots_alloc->ptr());

  auto d_info_alloc =
      phi::memory_utils::Alloc(dev_ctx.GetPlace(),
                               static_cast<size_t>(batch_count) * sizeof(int),
                               stream);
  int* d_info = reinterpret_cast<int*>(d_info_alloc->ptr());

  //    A_row layout: row-major (order x order), B_row layout: row-major (order
  //    x rhs_cols)
  for (int i = 0; i < batch_count; ++i) {
    const cuDoubleComplex* A_row =
        A_row_all + static_cast<size_t>(i) * order * order;
    cuDoubleComplex* A_col = dA_col + static_cast<size_t>(i) * order * order;
    const cuDoubleComplex* B_row =
        B_row_all + static_cast<size_t>(i) * order * rhs_cols;
    cuDoubleComplex* B_col = dB_col + static_cast<size_t>(i) * order * rhs_cols;

    // transpose A_row (row-major) -> A_col (column-major) via C = A^T
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasZgeam(cublas_handle,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             order,
                             order,
                             &kAlpha,
                             A_row,
                             order,  // lda: when interpreting A_row as (order x
                                     // order) row-major, using order
                             &kZero,
                             nullptr,
                             order,
                             A_col,
                             order));  // ldc = order (column-major leading dim)

    // transpose B_row (row-major order x rhs_cols) -> B_col (column-major order
    // x rhs_cols)
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasZgeam(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        order,
        rhs_cols,
        &kAlpha,
        B_row,
        rhs_cols,  // lda when A_row is viewed row-major: leading = rhs_cols
        &kZero,
        nullptr,
        rhs_cols,
        B_col,
        order));  // ldc = order
  }

  int lwork = 0;
  cuDoubleComplex* dA_col0 = dA_col;
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnZgetrf_bufferSize(
      cusolver_handle, order, order, dA_col0, order, &lwork));

  size_t work_bytes = static_cast<size_t>(lwork) * sizeof(cuDoubleComplex);
  auto d_work_alloc =
      phi::memory_utils::Alloc(dev_ctx.GetPlace(), work_bytes, stream);
  cuDoubleComplex* d_work =
      reinterpret_cast<cuDoubleComplex*>(d_work_alloc->ptr());

  for (int i = 0; i < batch_count; ++i) {
    cuDoubleComplex* A_col = dA_col + static_cast<size_t>(i) * order * order;
    cuDoubleComplex* B_col = dB_col + static_cast<size_t>(i) * order * rhs_cols;
    int* pivots_i = d_pivots + static_cast<size_t>(i) * order;
    int* info_i = d_info + i;

    // getrf (LU factorization) on A_col (column-major)
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnZgetrf(
        cusolver_handle, order, order, A_col, order, d_work, pivots_i, info_i));

    // getrs: solve A_col * X_col = B_col
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnZgetrs(
        cusolver_handle,
        CUBLAS_OP_N,  // no transpose on column-major matrix
        order,
        rhs_cols,
        A_col,
        order,
        pivots_i,
        B_col,
        order,
        info_i));
  }

  for (int i = 0; i < batch_count; ++i) {
    cuDoubleComplex* B_col = dB_col + static_cast<size_t>(i) * order *
                                          rhs_cols;  // X in column-major
    cuDoubleComplex* X_row = X_row_all + static_cast<size_t>(i) * order *
                                             rhs_cols;  // target row-major

    // transpose X_col -> X_row
    // We use C = A^T : A has shape (order x rhs_cols) in column-major, so C
    // will be (rhs_cols x order), but we want X_row with shape (order x
    // rhs_cols) in row-major; calling cublasZgeam with op=T and adjusted dims
    // works:
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasZgeam(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        rhs_cols,
        order,  // rowsC = rhs_cols, colsC = order
        &kAlpha,
        B_col,
        order,  // B_col lda = order (col-major)
        &kZero,
        nullptr,
        order,
        X_row,
        rhs_cols));  // X_row ldc = rhs_cols (row-major leading dimension)
  }

  std::vector<int> h_info(batch_count, 0);
  phi::memory_utils::Copy(CPUPlace(),
                          h_info.data(),
                          dev_ctx.GetPlace(),
                          d_info,
                          static_cast<size_t>(batch_count) * sizeof(int),
                          reinterpret_cast<void*>(dev_ctx.stream()));
  dev_ctx.Wait();

  for (int i = 0; i < batch_count; ++i) {
    PADDLE_ENFORCE_EQ(
        h_info[i],
        0,
        errors::External(
            "cuSOLVER getrf/getrs failed at batch %d, info: %d", i, h_info[i]));
  }
}
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
template <>
void SolveLinearSystemGPU<phi::dtype::complex<float>>(
    const GPUContext& dev_ctx,
    const phi::dtype::complex<float>*
        matrix_data,  // device ptr, row-major, size batch*order*order
    const phi::dtype::complex<float>*
        rhs_data,  // device ptr, row-major, size batch*order*rhs_cols
    phi::dtype::complex<float>*
        out_data,  // device ptr, row-major, size batch*order*rhs_cols
    int order,
    int rhs_cols,
    int batch_count) {
  // handles
  rocblas_handle rocblas_handle = dev_ctx.cusolver_dn_handle();
  auto stream = phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream()));

  // rocblas_float_complex constants
  const rocblas_float_complex kAlpha = rocblas_float_complex{1.0f, 0.0f};
  const rocblas_float_complex kZero = rocblas_float_complex{0.0f, 0.0f};

  // Sizes
  const size_t A_one_bytes =
      static_cast<size_t>(order) * order * sizeof(rocblas_float_complex);
  const size_t B_one_bytes =
      static_cast<size_t>(order) * rhs_cols * sizeof(rocblas_float_complex);
  const size_t A_batch_bytes = A_one_bytes * batch_count;
  const size_t B_batch_bytes = B_one_bytes * batch_count;

  const rocblas_float_complex* A_row_all =
      reinterpret_cast<const rocblas_float_complex*>(matrix_data);
  const rocblas_float_complex* B_row_all =
      reinterpret_cast<const rocblas_float_complex*>(rhs_data);
  rocblas_float_complex* X_row_all =
      reinterpret_cast<rocblas_float_complex*>(out_data);

  auto dA_col_alloc =
      phi::memory_utils::Alloc(dev_ctx.GetPlace(), A_batch_bytes, stream);
  auto dB_col_alloc =
      phi::memory_utils::Alloc(dev_ctx.GetPlace(), B_batch_bytes, stream);
  rocblas_float_complex* dA_col =
      reinterpret_cast<rocblas_float_complex*>(dA_col_alloc->ptr());
  rocblas_float_complex* dB_col =
      reinterpret_cast<rocblas_float_complex*>(dB_col_alloc->ptr());

  auto d_pivots_alloc = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      static_cast<size_t>(batch_count) * order * sizeof(rocblas_int),
      stream);
  rocblas_int* d_pivots = reinterpret_cast<rocblas_int*>(d_pivots_alloc->ptr());

  auto d_info_alloc = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      static_cast<size_t>(batch_count) * sizeof(rocblas_int),
      stream);
  rocblas_int* d_info = reinterpret_cast<rocblas_int*>(d_info_alloc->ptr());

  // A_row layout: row-major (order x order), B_row layout: row-major (order x
  // rhs_cols)
  for (int i = 0; i < batch_count; ++i) {
    const rocblas_float_complex* A_row =
        A_row_all + static_cast<size_t>(i) * order * order;
    rocblas_float_complex* A_col =
        dA_col + static_cast<size_t>(i) * order * order;
    const rocblas_float_complex* B_row =
        B_row_all + static_cast<size_t>(i) * order * rhs_cols;
    rocblas_float_complex* B_col =
        dB_col + static_cast<size_t>(i) * order * rhs_cols;

    // transpose A_row (row-major) -> A_col (column-major) via C = A^T
    PADDLE_ENFORCE_GPU_SUCCESS(rocblas_cgeam(
        rocblas_handle,
        rocblas_operation_transpose,
        rocblas_operation_none,
        order,
        order,
        &kAlpha,
        A_row,
        order,  // lda: when interpreting A_row as (order x order) row-major
        &kZero,
        nullptr,
        order,
        A_col,
        order));  // ldc = order (column-major leading dim)

    // transpose B_row (row-major order x rhs_cols) -> B_col (column-major order
    // x rhs_cols)
    PADDLE_ENFORCE_GPU_SUCCESS(rocblas_cgeam(
        rocblas_handle,
        rocblas_operation_transpose,
        rocblas_operation_none,
        order,
        rhs_cols,
        &kAlpha,
        B_row,
        rhs_cols,  // lda when A_row is viewed row-major: leading = rhs_cols
        &kZero,
        nullptr,
        rhs_cols,
        B_col,
        order));  // ldc = order
  }

  // LU factorization and solve for each batch
  for (int i = 0; i < batch_count; ++i) {
    rocblas_float_complex* A_col =
        dA_col + static_cast<size_t>(i) * order * order;
    rocblas_float_complex* B_col =
        dB_col + static_cast<size_t>(i) * order * rhs_cols;
    rocblas_int* pivots_i = d_pivots + static_cast<size_t>(i) * order;
    rocblas_int* info_i = d_info + i;

    // getrf (LU factorization) on A_col (column-major)
    PADDLE_ENFORCE_GPU_SUCCESS(rocsolver_cgetrf(
        rocblas_handle, order, order, A_col, order, pivots_i, info_i));

    // getrs: solve A_col * X_col = B_col
    PADDLE_ENFORCE_GPU_SUCCESS(rocsolver_cgetrs(
        rocblas_handle,
        rocblas_operation_none,  // no transpose on column-major matrix
        order,
        rhs_cols,
        A_col,
        order,
        pivots_i,
        B_col,
        order));
  }

  // Transpose results back to row-major
  for (int i = 0; i < batch_count; ++i) {
    rocblas_float_complex* B_col = dB_col + static_cast<size_t>(i) * order *
                                                rhs_cols;  // X in column-major
    rocblas_float_complex* X_row =
        X_row_all +
        static_cast<size_t>(i) * order * rhs_cols;  // target row-major

    // transpose X_col -> X_row
    PADDLE_ENFORCE_GPU_SUCCESS(rocblas_cgeam(
        rocblas_handle,
        rocblas_operation_transpose,
        rocblas_operation_none,
        rhs_cols,
        order,  // rowsC = rhs_cols, colsC = order
        &kAlpha,
        B_col,
        order,  // B_col lda = order (col-major)
        &kZero,
        nullptr,
        order,
        X_row,
        rhs_cols));  // X_row ldc = rhs_cols (row-major leading dimension)
  }

  // Check error info
  CPUPlace cpu_place;
  DeviceContextPool& pool = DeviceContextPool::Instance();
  auto* cpu_ctx = static_cast<CPUContext*>(pool.Get(cpu_place));

  std::vector<rocblas_int> h_info(batch_count, 0);
  phi::memory_utils::Copy(
      CPUPlace(),
      h_info.data(),
      dev_ctx.GetPlace(),
      d_info,
      static_cast<size_t>(batch_count) * sizeof(rocblas_int),
      reinterpret_cast<void*>(dev_ctx.stream()));
  dev_ctx.Wait();

  for (int i = 0; i < batch_count; ++i) {
    PADDLE_ENFORCE_EQ(
        h_info[i],
        0,
        errors::External("rocSOLVER getrf/getrs failed at batch %d, info: %d",
                         i,
                         h_info[i]));
  }
}

template <>
void SolveLinearSystemGPU<phi::dtype::complex<double>>(
    const GPUContext& dev_ctx,
    const phi::dtype::complex<double>*
        matrix_data,  // device ptr, row-major, size batch*order*order
    const phi::dtype::complex<double>*
        rhs_data,  // device ptr, row-major, size batch*order*rhs_cols
    phi::dtype::complex<double>*
        out_data,  // device ptr, row-major, size batch*order*rhs_cols
    int order,
    int rhs_cols,
    int batch_count) {
  // handles
  rocblas_handle rocblas_handle = dev_ctx.cusolver_dn_handle();
  auto stream = phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream()));

  // rocblas_double_complex constants
  const rocblas_double_complex kAlpha = rocblas_double_complex{1.0, 0.0};
  const rocblas_double_complex kZero = rocblas_double_complex{0.0, 0.0};

  // Sizes
  const size_t A_one_bytes =
      static_cast<size_t>(order) * order * sizeof(rocblas_double_complex);
  const size_t B_one_bytes =
      static_cast<size_t>(order) * rhs_cols * sizeof(rocblas_double_complex);
  const size_t A_batch_bytes = A_one_bytes * batch_count;
  const size_t B_batch_bytes = B_one_bytes * batch_count;

  const rocblas_double_complex* A_row_all =
      reinterpret_cast<const rocblas_double_complex*>(matrix_data);
  const rocblas_double_complex* B_row_all =
      reinterpret_cast<const rocblas_double_complex*>(rhs_data);
  rocblas_double_complex* X_row_all =
      reinterpret_cast<rocblas_double_complex*>(out_data);

  auto dA_col_alloc =
      phi::memory_utils::Alloc(dev_ctx.GetPlace(), A_batch_bytes, stream);
  auto dB_col_alloc =
      phi::memory_utils::Alloc(dev_ctx.GetPlace(), B_batch_bytes, stream);
  rocblas_double_complex* dA_col =
      reinterpret_cast<rocblas_double_complex*>(dA_col_alloc->ptr());
  rocblas_double_complex* dB_col =
      reinterpret_cast<rocblas_double_complex*>(dB_col_alloc->ptr());

  auto d_pivots_alloc = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      static_cast<size_t>(batch_count) * order * sizeof(rocblas_int),
      stream);
  rocblas_int* d_pivots = reinterpret_cast<rocblas_int*>(d_pivots_alloc->ptr());

  auto d_info_alloc = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      static_cast<size_t>(batch_count) * sizeof(rocblas_int),
      stream);
  rocblas_int* d_info = reinterpret_cast<rocblas_int*>(d_info_alloc->ptr());

  // A_row layout: row-major (order x order), B_row layout: row-major (order x
  // rhs_cols)
  for (int i = 0; i < batch_count; ++i) {
    const rocblas_double_complex* A_row =
        A_row_all + static_cast<size_t>(i) * order * order;
    rocblas_double_complex* A_col =
        dA_col + static_cast<size_t>(i) * order * order;
    const rocblas_double_complex* B_row =
        B_row_all + static_cast<size_t>(i) * order * rhs_cols;
    rocblas_double_complex* B_col =
        dB_col + static_cast<size_t>(i) * order * rhs_cols;

    // transpose A_row (row-major) -> A_col (column-major) via C = A^T
    PADDLE_ENFORCE_GPU_SUCCESS(rocblas_zgeam(
        rocblas_handle,
        rocblas_operation_transpose,
        rocblas_operation_none,
        order,
        order,
        &kAlpha,
        A_row,
        order,  // lda: when interpreting A_row as (order x order) row-major
        &kZero,
        nullptr,
        order,
        A_col,
        order));  // ldc = order (column-major leading dim)

    // transpose B_row (row-major order x rhs_cols) -> B_col (column-major order
    // x rhs_cols)
    PADDLE_ENFORCE_GPU_SUCCESS(rocblas_zgeam(
        rocblas_handle,
        rocblas_operation_transpose,
        rocblas_operation_none,
        order,
        rhs_cols,
        &kAlpha,
        B_row,
        rhs_cols,  // lda when A_row is viewed row-major: leading = rhs_cols
        &kZero,
        nullptr,
        rhs_cols,
        B_col,
        order));  // ldc = order
  }

  // LU factorization and solve for each batch
  for (int i = 0; i < batch_count; ++i) {
    rocblas_double_complex* A_col =
        dA_col + static_cast<size_t>(i) * order * order;
    rocblas_double_complex* B_col =
        dB_col + static_cast<size_t>(i) * order * rhs_cols;
    rocblas_int* pivots_i = d_pivots + static_cast<size_t>(i) * order;
    rocblas_int* info_i = d_info + i;

    // getrf (LU factorization) on A_col (column-major)
    PADDLE_ENFORCE_GPU_SUCCESS(rocsolver_zgetrf(
        rocblas_handle, order, order, A_col, order, pivots_i, info_i));

    // getrs: solve A_col * X_col = B_col
    PADDLE_ENFORCE_GPU_SUCCESS(rocsolver_zgetrs(
        rocblas_handle,
        rocblas_operation_none,  // no transpose on column-major matrix
        order,
        rhs_cols,
        A_col,
        order,
        pivots_i,
        B_col,
        order));
  }

  // Transpose results back to row-major
  for (int i = 0; i < batch_count; ++i) {
    rocblas_double_complex* B_col = dB_col + static_cast<size_t>(i) * order *
                                                 rhs_cols;  // X in column-major
    rocblas_double_complex* X_row =
        X_row_all +
        static_cast<size_t>(i) * order * rhs_cols;  // target row-major

    // transpose X_col -> X_row
    PADDLE_ENFORCE_GPU_SUCCESS(rocblas_zgeam(
        rocblas_handle,
        rocblas_operation_transpose,
        rocblas_operation_none,
        rhs_cols,
        order,  // rowsC = rhs_cols, colsC = order
        &kAlpha,
        B_col,
        order,  // B_col lda = order (col-major)
        &kZero,
        nullptr,
        order,
        X_row,
        rhs_cols));  // X_row ldc = rhs_cols (row-major leading dimension)
  }
  CPUPlace cpu_place;
  DeviceContextPool& pool = DeviceContextPool::Instance();
  auto* cpu_ctx = static_cast<CPUContext*>(pool.Get(cpu_place));

  std::vector<rocblas_int> h_info(batch_count, 0);
  phi::memory_utils::Copy(
      CPUPlace(),
      h_info.data(),
      dev_ctx.GetPlace(),
      d_info,
      static_cast<size_t>(batch_count) * sizeof(rocblas_int),
      reinterpret_cast<void*>(dev_ctx.stream()));
  dev_ctx.Wait();

  for (int i = 0; i < batch_count; ++i) {
    PADDLE_ENFORCE_EQ(
        h_info[i],
        0,
        errors::External("rocSOLVER getrf/getrs failed at batch %d, info: %d",
                         i,
                         h_info[i]));
  }
}
#endif  // PADDLE_WITH_HIP

template <typename T, typename Context>
void ComputeBackwardForComplexInputGPU(const DenseTensor& L,
                                       const DenseTensor& V,
                                       const optional<DenseTensor>& gL,
                                       const optional<DenseTensor>& gV,
                                       T* x_grad_data,
                                       int batch_count,
                                       int order,
                                       const Context& dev_ctx) {
  DenseTensor gL_safe;
  if (gL.get_ptr()) {
    gL_safe = gL.get();
  } else {
    gL_safe = Fill<T, Context>(dev_ctx, vectorize<int64_t>(L.dims()), T(0));
  }

  DenseTensor gV_safe;
  if (gV.get_ptr()) {
    gV_safe = gV.get();
  } else {
    gV_safe = Fill<T, Context>(dev_ctx, vectorize<int64_t>(V.dims()), T(0));
  }
  DenseTensor trans_v = TransposeLast2Dim<T>(dev_ctx, V);
  DenseTensor Vh = phi::Conj<T>(dev_ctx, trans_v);
  DenseTensor Lconj = phi::Conj<T>(dev_ctx, L);
  DenseTensor Econj = phi::Subtract<T>(dev_ctx,
                                       phi::funcs::Unsqueeze(Lconj, -2),
                                       phi::funcs::Unsqueeze(Lconj, -1));
  DenseTensor VhgV = phi::Matmul<T>(dev_ctx, Vh, gV_safe);
  DenseTensor diag_real = phi::Real<T>(dev_ctx, VhgV);

  auto cpu_place = CPUPlace();
  DeviceContextPool& pool = DeviceContextPool::Instance();
  auto* cpu_ctx = static_cast<CPUContext*>(pool.Get(cpu_place));

  DenseTensor diag_real_cpu;
  diag_real_cpu.Resize(diag_real.dims());
  Copy(dev_ctx, diag_real, cpu_place, false, &diag_real_cpu);

  DenseTensor diag_res_cpu =
      phi::funcs::BatchDiag<T>((*cpu_ctx), diag_real_cpu, batch_count);

  DenseTensor diag_res;
  dev_ctx.template Alloc<T>(&diag_res);
  Copy(dev_ctx, diag_res_cpu, GPUPlace(), false, &diag_res);

  DenseTensor diag_unsqueezed = phi::funcs::Unsqueeze(diag_res, -2);

  auto numel = diag_unsqueezed.numel();
  DenseTensor diag_unsqueezed_complex;
  auto* data_diag_un = diag_unsqueezed.data<dtype::Real<T>>();
  diag_unsqueezed_complex.Resize(diag_unsqueezed.dims());
  auto* data_diag_un_com = dev_ctx.template Alloc<T>(
      &diag_unsqueezed_complex, static_cast<size_t>(numel * sizeof(T)));

  phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
  phi::funcs::RealToComplexFunctor<T> functor(
      data_diag_un, data_diag_un_com, numel);
  for_range(functor);
  // real tensor multiply complex tensor in broadcast manner
  DenseTensor res1 = phi::Multiply<T>(dev_ctx, V, diag_unsqueezed_complex);
  DenseTensor res2 = phi::Matmul<T>(dev_ctx, Vh, res1);
  DenseTensor result = phi::Subtract<T>(dev_ctx, VhgV, res2);

  result.Resize(V.dims());
  dev_ctx.template Alloc<T>(&result);
  result = phi::Divide<T>(dev_ctx, result, Econj);
  result = phi::funcs::DiagFill<T, T>(
      dev_ctx, order, order, order, 0, gL_safe, result);
  DenseTensor rhs = phi::Matmul<T>(dev_ctx, result, Vh);

  // solve linear system
  // solve(Vh, rhs, out, m, k)
  // Vh: matrix with shape [m,m]
  // rhs: rhs with shape [m,k]
  // x_grad: out
  int64_t m = Vh.dims(-1);
  int64_t k = rhs.dims(-1);
  auto* matrix_data = Vh.data<T>();
  auto* rhs_data = rhs.data<T>();

  SolveLinearSystemGPU<T>(
      dev_ctx, matrix_data, rhs_data, x_grad_data, m, k, batch_count);
}
#endif  // PADDLE_WITH_MAGMA

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename T, typename Context>
void EigGradKernel(const Context& dev_ctx,
                   const DenseTensor& out_w,
                   const DenseTensor& out_v,
                   const optional<DenseTensor>& dout_w,
                   const optional<DenseTensor>& dout_v,
                   DenseTensor* dx) {
  auto* dx_data = dev_ctx.template Alloc<phi::dtype::Complex<T>>(dx);
  if (dx->numel() == 0) {
    return;
  }
  auto& dims = out_v.dims();
  int batch_count = BatchCount(out_v);
  const int64_t order = out_v.dims(-1);

  ComputeBackwardForComplexInputGPU<phi::dtype::Complex<T>, Context>(
      out_w, out_v, dout_w, dout_v, dx_data, batch_count, order, dev_ctx);
}
#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP

}  // namespace phi

// Register the kernel
#ifdef PADDLE_WITH_MAGMA
PD_REGISTER_KERNEL(eig_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::EigGradKernel,
                   float,
                   double,
                   phi::complex64,
                   phi::complex128) {
  kernel->InputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
  kernel->InputAt(2).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
  kernel->OutputAt(0).SetDataType(phi::dtype::ToComplex(kernel_key.dtype()));
}
#endif  // PADDLE_WITH_MAGMA
