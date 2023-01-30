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

#pragma once
#ifdef PADDLE_WITH_CUDA

#include <cuda_runtime_api.h>
#include "cuda.h"  // NOLINT

#if CUDA_VERSION >= 11060
#include "paddle/phi/kernels/autotune/cache_cublas_Lt.h"

namespace phi {

// template <typename T, class Context>
// struct CublasLtGEMM {
//   void operator()(const Context& dev_ctx,
//                   const T* x_data,
//                   const T* y_data,
//                   const int M,
//                   const int N,
//                   const int K,
//                   T* out_data,
//                   bool trans_x,
//                   bool trans_y) {}
// };

// template <typename T, class Context>
// struct CublasLtBatchedGEMM {
//   void operator()(const Context& dev_ctx,
//                   const T* x_data,
//                   const T* y_data,
//                   const int M,
//                   const int N,
//                   const int K,
//                   T* out_data,
//                   bool trans_x,
//                   bool trans_y,
//                   int batch_size,
//                   int64_t stride_x,
//                   int64_t stride_y,
//                   int64_t stride_out) {}
//   void operator()(const Context& dev_ctx,
//                   const T** x_data,
//                   const T** y_data,
//                   const int M,
//                   const int N,
//                   const int K,
//                   T** out_data,
//                   bool trans_x,
//                   bool trans_y,
//                   int batch_size) {}
// };

// template <typename T>
// struct TypeTrait {
//   cudaDataType_t mat_type = CUDA_R_32F;
//   cudaDataType_t scale_type = CUDA_R_32F;
//   cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
// };

// template <>
// struct TypeTrait<phi::dtype::float16> {
//   cudaDataType_t mat_type = CUDA_R_16F;
//   cudaDataType_t scale_type = CUDA_R_32F;
//   cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
// };

// template <>
// struct TypeTrait<phi::dtype::bfloat16> {
//   cudaDataType_t mat_type = CUDA_R_16BF;
//   cudaDataType_t scale_type = CUDA_R_32F;
//   cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
// };

// template <>
// struct TypeTrait<double> {
//   cudaDataType_t mat_type = CUDA_R_64F;
//   cudaDataType_t scale_type = CUDA_R_64F;
//   cublasComputeType_t compute_type = CUBLAS_COMPUTE_64F;
// };

// template <typename T>
// struct CublasLtGEMM<T, phi::GPUContext> {
//   void operator()(const phi::GPUContext& dev_ctx,
//                   const T* x_data,
//                   const T* y_data,
//                   const int M,
//                   const int N,
//                   const int K,
//                   T* out_data,
//                   bool trans_x,
//                   bool trans_y) {
//     // init data structure
//     cublasLtHandle_t lt_handle = dev_ctx.cublaslt_handle();

//     cublasLtMatmulDesc_t operation_desc = NULL;
//     cublasLtMatrixLayout_t x_desc = NULL, y_desc = NULL, out_desc = NULL;

//     cublasOperation_t transx = trans_x ? CUBLAS_OP_T : CUBLAS_OP_N;
//     cublasOperation_t transy = trans_y ? CUBLAS_OP_T : CUBLAS_OP_N;

//     TypeTrait<T> MT;
//     cudaDataType_t mat_type = MT.mat_type;
//     cudaDataType_t scale_type = MT.scale_type;
//     cublasComputeType_t compute_type = MT.compute_type;

//     // Create operation desciriptor; see cublasLtMatmulDescAttributes_t for
//     // details about defaults; This OP we just need to set the transforms for
//     A
//     // and B
//     PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmulDescCreate(
//         &operation_desc, compute_type, scale_type));
//     PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmulDescSetAttribute(
//         operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transx,
//         sizeof(transx)));
//     PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmulDescSetAttribute(
//         operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transy,
//         sizeof(transy)));

//     // Create matrix descriptors
//     if (trans_x)
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           phi::dynload::cublasLtMatrixLayoutCreate(&x_desc, mat_type, M, K,
//           M));
//     else
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           phi::dynload::cublasLtMatrixLayoutCreate(&x_desc, mat_type, K, M,
//           K));
//     if (trans_y)
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           phi::dynload::cublasLtMatrixLayoutCreate(&y_desc, mat_type, K, N,
//           K));
//     else
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           phi::dynload::cublasLtMatrixLayoutCreate(&y_desc, mat_type, N, K,
//           N));

//     PADDLE_ENFORCE_GPU_SUCCESS(
//         phi::dynload::cublasLtMatrixLayoutCreate(&out_desc, mat_type, N, M,
//         N));

//     double alpha64 = 1.0, beta64 = 0.0;
//     float alpha32 = 1.0f, beta32 = 0.0f;
//     void *alpha = nullptr, *beta = nullptr;
//     if (std::is_same<T, double>::value) {
//       alpha = &alpha64;
//       alpha = &beta64;
//     } else {
//       alpha = &alpha32;
//       beta = &beta32;
//     }

//     size_t workspace_size = static_cast<size_t>(4) * 1024 * 1024;
//     cudaStream_t stream = dev_ctx.stream();
//     phi::Allocator::AllocationPtr workspace = paddle::memory::Alloc(
//         dev_ctx.GetPlace(),
//         workspace_size,
//         phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

//     auto algo = CublasLtAlgoCache::Instance().GetGemmAlgo(lt_handle,
//                                                           operation_desc,
//                                                           y_desc,
//                                                           x_desc,
//                                                           out_desc,
//                                                           alpha,
//                                                           beta,
//                                                           y_data,
//                                                           x_data,
//                                                           out_data,
//                                                           stream,
//                                                           workspace->ptr(),
//                                                           workspace_size);
//     // We can take the advantage of cublasLtMatmul shortcut notation with
//     // algo = NULL which will force matmul to get the basic heuristic result
//     // internally. Downsides of this approach are that there is no way to
//     // configure search preferences (e.g. disallow tensor operations or some
//     // reduction schemes) and no way to store the algo for later use
//     PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmul(lt_handle,
//                                                             operation_desc,
//                                                             alpha,
//                                                             y_data,
//                                                             y_desc,
//                                                             x_data,
//                                                             x_desc,
//                                                             beta,
//                                                             out_data,
//                                                             out_desc,
//                                                             out_data,
//                                                             out_desc,
//                                                             algo,
//                                                             workspace->ptr(),
//                                                             workspace_size,
//                                                             stream));
//     // Descriptors are no longer needed as all GPU work was already enqueued
//     if (y_desc)
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           phi::dynload::cublasLtMatrixLayoutDestroy(y_desc));
//     if (x_desc)
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           phi::dynload::cublasLtMatrixLayoutDestroy(x_desc));
//     if (out_desc)
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           phi::dynload::cublasLtMatrixLayoutDestroy(out_desc));
//     if (operation_desc)
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           phi::dynload::cublasLtMatmulDescDestroy(operation_desc));
//     return;
//   }
// };

// template <typename T>
// struct CublasLtBatchedGEMM<T, phi::GPUContext> {
//   void operator()(const phi::GPUContext& dev_ctx,
//                   const T* x_data,
//                   const T* y_data,
//                   const int M,
//                   const int N,
//                   const int K,
//                   T* out_data,
//                   bool trans_x,
//                   bool trans_y,
//                   int batch_size,
//                   int64_t stride_x,
//                   int64_t stride_y,
//                   int64_t stride_out) {
//     // init data structure
//     cublasLtHandle_t lt_handle = dev_ctx.cublaslt_handle();

//     cublasLtMatmulDesc_t operation_desc = NULL;
//     cublasLtMatrixLayout_t x_desc = NULL, y_desc = NULL, out_desc = NULL;

//     cublasOperation_t transx = trans_x ? CUBLAS_OP_T : CUBLAS_OP_N;
//     cublasOperation_t transy = trans_y ? CUBLAS_OP_T : CUBLAS_OP_N;

//     TypeTrait<T> MT;
//     cudaDataType_t mat_type = MT.mat_type;
//     cudaDataType_t scale_type = MT.scale_type;
//     cublasComputeType_t compute_type = MT.compute_type;

//     // Create operation desciriptor; see cublasLtMatmulDescAttributes_t for
//     // details about defaults; This OP we just need to set the transforms for
//     A
//     // and B
//     PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmulDescCreate(
//         &operation_desc, compute_type, scale_type));
//     PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmulDescSetAttribute(
//         operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transx,
//         sizeof(transx)));
//     PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmulDescSetAttribute(
//         operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transy,
//         sizeof(transy)));

//     // Create matrix descriptors, this case batch size and counts should be
//     // configured
//     if (trans_x)
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           phi::dynload::cublasLtMatrixLayoutCreate(&x_desc, mat_type, M, K,
//           M));
//     else
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           phi::dynload::cublasLtMatrixLayoutCreate(&x_desc, mat_type, K, M,
//           K));
//     if (trans_y)
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           phi::dynload::cublasLtMatrixLayoutCreate(&y_desc, mat_type, K, N,
//           K));
//     else
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           phi::dynload::cublasLtMatrixLayoutCreate(&y_desc, mat_type, N, K,
//           N));

//     PADDLE_ENFORCE_GPU_SUCCESS(
//         phi::dynload::cublasLtMatrixLayoutCreate(&out_desc, mat_type, N, M,
//         N));

//     // Config batch size and counts
//     PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatrixLayoutSetAttribute(
//         x_desc,
//         CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
//         &batch_size,
//         sizeof(batch_size)));
//     PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatrixLayoutSetAttribute(
//         x_desc,
//         CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
//         &stride_x,
//         sizeof(stride_x)));

//     PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatrixLayoutSetAttribute(
//         y_desc,
//         CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
//         &batch_size,
//         sizeof(batch_size)));
//     PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatrixLayoutSetAttribute(
//         y_desc,
//         CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
//         &stride_y,
//         sizeof(stride_y)));

//     PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatrixLayoutSetAttribute(
//         out_desc,
//         CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
//         &batch_size,
//         sizeof(batch_size)));
//     PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatrixLayoutSetAttribute(
//         out_desc,
//         CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
//         &stride_out,
//         sizeof(stride_out)));

//     double alpha64 = 1.0, beta64 = 0.0;
//     float alpha32 = 1.0f, beta32 = 0.0f;
//     void *alpha = nullptr, *beta = nullptr;
//     if (std::is_same<T, double>::value) {
//       alpha = &alpha64;
//       alpha = &beta64;
//     } else {
//       alpha = &alpha32;
//       beta = &beta32;
//     }

//     size_t workspace_size = static_cast<size_t>(4) * 1024 * 1024;
//     cudaStream_t stream = dev_ctx.stream();
//     phi::Allocator::AllocationPtr workspace = paddle::memory::Alloc(
//         dev_ctx.GetPlace(),
//         workspace_size,
//         phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

//     auto algo = CublasLtAlgoCache::Instance().GetGemmAlgo(lt_handle,  // OK
//                                                           operation_desc,
//                                                           y_desc,
//                                                           x_desc,
//                                                           out_desc,
//                                                           alpha,
//                                                           beta,
//                                                           y_data,
//                                                           x_data,
//                                                           out_data,
//                                                           stream,
//                                                           workspace->ptr(),
//                                                           workspace_size);
//     // We can take the advantage of cublasLtMatmul shortcut notation with
//     // algo = NULL which will force matmul to get the basic heuristic result
//     // internally. Downsides of this approach are that there is no way to
//     // configure search preferences (e.g. disallow tensor operations or some
//     // reduction schemes) and no way to store the algo for later use
//     PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmul(lt_handle,
//                                                             operation_desc,
//                                                             alpha,
//                                                             y_data,
//                                                             y_desc,
//                                                             x_data,
//                                                             x_desc,
//                                                             beta,
//                                                             out_data,
//                                                             out_desc,
//                                                             out_data,
//                                                             out_desc,
//                                                             algo,
//                                                             workspace->ptr(),
//                                                             workspace_size,
//                                                             stream));
//     // Descriptors are no longer needed as all GPU work was already enqueued
//     if (y_desc)
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           phi::dynload::cublasLtMatrixLayoutDestroy(y_desc));
//     if (x_desc)
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           phi::dynload::cublasLtMatrixLayoutDestroy(x_desc));
//     if (out_desc)
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           phi::dynload::cublasLtMatrixLayoutDestroy(out_desc));
//     if (operation_desc)
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           phi::dynload::cublasLtMatmulDescDestroy(operation_desc));
//     return;
//   }
//   void operator()(const phi::GPUContext& dev_ctx,
//                   const T** x_data,
//                   const T** y_data,
//                   const int M,
//                   const int N,
//                   const int K,
//                   T** out_data,
//                   bool trans_x,
//                   bool trans_y,
//                   int batch_size) {
//     for (int k = 0; k < batch_size; ++k) {
//       CublasLtGEMM<T, phi::GPUContext>()(dev_ctx,
//                                          x_data[k],
//                                          y_data[k],
//                                          M,
//                                          N,
//                                          K,
//                                          out_data[k],
//                                          trans_x,
//                                          trans_y);
//     }
//   }
// };

}  // namespace phi

#endif
#endif
