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
#include "paddle/phi/kernels/autotune/gpu_timer.h"
#include "paddle/phi/kernels/autotune/switch_autotune.h"

#if CUDA_VERSION >= 11060
#include "paddle/phi/kernels/autotune/cache_cublas_Lt.h"
#endif

namespace phi {

struct cublasLtMatmulAlgoTypeImitater {
  int64_t data[8];
}

#ifdef PADDLE_WITH_CUDA
// template <typename T, class Context>
// struct CublasLtGEMM {
//   void operator()(const Context& dev_ctx,
//                   const T* x_data,
//                   const T* y_data,
//                   T* out_data,
//                   const int M,
//                   const int N,
//                   const int K,
//                   bool trans_x,
//                   bool trans_y) {}
// };

// template <typename T, class Context>
// struct CublasLtBatchedGEMM {
//   void operator()(const Context& dev_ctx,
//                   const T* x_data,
//                   const T* y_data,
//                   T* out_data,
//                   const int M,
//                   const int N,
//                   const int K,
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
// struct CublasLtBatchedGEMM<T, phi::GPUContext> {
//   void operator()(const phi::GPUContext& dev_ctx,
//                   const T* x_data,
//                   const T* y_data,
//                   T* out_data,
//                   const int M,
//                   const int N,
//                   const int K,
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
//     // details about defaults; just need to set the transforms for A and B
//     PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescCreate(
//         &operation_desc, compute_type, scale_type));
//     PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescSetAttribute(
//         operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transx, sizeof(transx)));
//     PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescSetAttribute(
//         operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transy, sizeof(transy)));

//     // Create matrix descriptors, this case batch size and counts should be
//     // configured
//     if (trans_x) {
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           dynload::cublasLtMatrixLayoutCreate(&x_desc, mat_type, M, K, M));
//     } else {
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           dynload::cublasLtMatrixLayoutCreate(&x_desc, mat_type, K, M, K));
//     }
//     if (trans_y) {
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           dynload::cublasLtMatrixLayoutCreate(&y_desc, mat_type, K, N, K));
//     }  else {
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           dynload::cublasLtMatrixLayoutCreate(&y_desc, mat_type, N, K, N));
//     }
//     PADDLE_ENFORCE_GPU_SUCCESS(
//         dynload::cublasLtMatrixLayoutCreate(&out_desc, mat_type, N, M, N));

//     // Config batch size and counts
//     PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutSetAttribute(
//         x_desc,
//         CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
//         &batch_size,
//         sizeof(batch_size)));
//     PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutSetAttribute(
//         x_desc,
//         CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
//         &stride_x,
//         sizeof(stride_x)));

//     PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutSetAttribute(
//         y_desc,
//         CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
//         &batch_size,
//         sizeof(batch_size)));
//     PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutSetAttribute(
//         y_desc,
//         CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
//         &stride_y,
//         sizeof(stride_y)));

//     PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutSetAttribute(
//         out_desc,
//         CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
//         &batch_size,
//         sizeof(batch_size)));
//     PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutSetAttribute(
//         out_desc,
//         CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
//         &stride_out,
//         sizeof(stride_out)));

//     double alpha64 = 1.0, beta64 = 0.0;
//     float alpha32 = 1.0f, beta32 = 0.0f;
//     void *alpha = std::is_same<T, double>::value ? &alpha64 : &alpha32;
//     void *beta = std::is_same<T, double>::value ? &beta64 : &beta32;

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
//     PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmul(lt_handle,
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
//           dynload::cublasLtMatrixLayoutDestroy(y_desc));
//     if (x_desc)
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           dynload::cublasLtMatrixLayoutDestroy(x_desc));
//     if (out_desc)
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           dynload::cublasLtMatrixLayoutDestroy(out_desc));
//     if (operation_desc)
//       PADDLE_ENFORCE_GPU_SUCCESS(
//           dynload::cublasLtMatmulDescDestroy(operation_desc));
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

template <typename T, class Enable = void>
struct MatmulCalculationClassifier {
  void operator()(cudaDataType_t* math_type, 
                  cudaDataType_t* scale_type, 
                  cublasComputeType_t* compute_type) {}
};

template <typename T>
struct MatmulCalculationClassifier<T, 
                          std::enable_if_t<std::is_same<T, phi::dtype::float16>::value>> {
  void operator()(cudaDataType_t* math_type, 
                  cudaDataType_t* scale_type, 
                  cublasComputeType_t* compute_type) {
    *mat_type = CUDA_R_16F;
  }
};

template <typename T>
struct MatmulCalculationClassifier<T, 
                          std::enable_if_t<std::is_same<T, phi::dtype::bfloat16>::value>> {
  void operator()(cudaDataType_t* math_type, 
                  cudaDataType_t* scale_type, 
                  cublasComputeType_t* compute_type) {
    *mat_type = CUDA_R_16BF;
  }
};

template <typename T>
struct MatmulCalculationClassifier<T, 
                          std::enable_if_t<std::is_same<T, double>::value>> {
  void operator()(cudaDataType_t* math_type, 
                  cudaDataType_t* scale_type, 
                  cublasComputeType_t* compute_type) {
    *mat_type = CUDA_R_64F;
    *scale_type = CUDA_R_64F;
    *compute_type = CUBLAS_COMPUTE_64F;
  }
};

struct MatmulDescCreator {
 public:
  static void Create(cublasLtMatmulDesc_t* op_desc,
                     cublasLtMatrixLayout_t* x_desc, 
                     cublasLtMatrixLayout_t* y_desc, 
                     cublasLtMatrixLayout_t* out_desc,
                     cublasComputeType_t compute_type,
                     cudaDataType_t mat_type,
                     cudaDataType_t scale_type,
                     const int M,
                     const int N,
                     const int K,
                     const bool trans_x,
                     const bool trans_y,
                     const int batch_size,
                     const int64_t stride_x,
                     const int64_t stride_y,
                     const int64_t stride_out) {
    // Create operation desciriptor; see cublasLtMatmulDescAttributes_t for
    // details about defaults; just need to set the transforms for A and B
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescCreate(
        op_desc, compute_type, scale_type));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescSetAttribute(
        *op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transx, sizeof(transx)));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescSetAttribute(
        *op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transy, sizeof(transy)));

    // Create matrix descriptors
    if (trans_x) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cublasLtMatrixLayoutCreate(x_desc, mat_type, M, K, M));
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cublasLtMatrixLayoutCreate(x_desc, mat_type, K, M, K));
    }
    if (trans_y) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cublasLtMatrixLayoutCreate(y_desc, mat_type, K, N, K));
    }  else {
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cublasLtMatrixLayoutCreate(y_desc, mat_type, N, K, N));
    }
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatrixLayoutCreate(out_desc, mat_type, N, M, N));

    // Config batch size and stride.
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutSetAttribute(
        *x_desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_size,
        sizeof(batch_size)));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutSetAttribute(
        *y_desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_size,
        sizeof(batch_size)));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutSetAttribute(
        *out_desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_size,
        sizeof(batch_size)));

    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutSetAttribute(
        *x_desc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_x,
        sizeof(stride_x)));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutSetAttribute(
        *y_desc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_y,
        sizeof(stride_y)));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutSetAttribute(
        *out_desc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_out,
        sizeof(stride_out)));
  }

  static void Release(cublasLtMatmulDesc_t* op_desc,
                      cublasLtMatrixLayout_t* x_desc,
                      cublasLtMatrixLayout_t* y_desc,
                      cublasLtMatrixLayout_t* out_desc) {
    if (*y_desc) {
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutDestroy(*y_desc));
    }
    if (*x_desc) {
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutDestroy(*x_desc));
    }
    if (*out_desc) {
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutDestroy(*out_desc));
    }
    if (*op_desc) {
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescDestroy(*op_desc));
    }
  }
}

template <typename T>
struct MatmulWithCublasLt {
 public:
  static void Run(const phi::GPUContext& ctx,
                  const AlgorithmType& algo,
                  const size_t best_idx,
                  const T* x_data,
                  const T* y_data,
                  T* out_data,
                  const int M,
                  const int N,
                  const int K,
                  const bool trans_x,
                  const bool trans_y,
                  const int batch_size,
                  const int64_t stride_x,
                  const int64_t stride_y,
                  const int64_t stride_out) {
    // init data structure
    cublasLtHandle_t lt_handle = ctx.cublaslt_handle();
    cublasLtMatmulDesc_t op_desc = NULL;
    cublasLtMatrixLayout_t x_desc = NULL, y_desc = NULL, out_desc = NULL;
    cublasOperation_t transx = trans_x ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transy = trans_y ? CUBLAS_OP_T : CUBLAS_OP_N;

    cudaDataType_t mat_type = CUDA_R_32F;
    cudaDataType_t scale_type = CUDA_R_32F;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    MatmulCalculationClassifier<T>()(&mat_type, &scale_type, &compute_type);
    MatmulDescCreator::Create(&op_desc, &x_desc, &y_desc, &out_desc, compute_type,
                              mat_type, scale_type, M, N, K, trans_x, trans_y,
                              batch_size, stride_x, stride_y, stride_out);

    float alpha32 = 1.0f, beta32 = 0.0f;
    double alpha64 = static_cast<double>(1);
    double beta64 = static_cast<double>(0);
    void *alpha = std::is_same<T, double>::value ? &alpha64 : &alpha32;
    void *beta = std::is_same<T, double>::value ? &beta64 : &beta32;
    size_t workspace_size = static_cast<size_t>(4) * 1024 * 1024; 
    phi::Allocator::AllocationPtr workspace = GetWorkspace(ctx, workspace_size);

    bool use_autotune = phi::autotune::AutoTuneStatus::Instance().UseAutoTune();
    auto& cache = AutoTuneCache::Instance().Get(algo);
    cublasLtMatmulAlgo_t* best_algo = 
              static_cast<cublasLtMatmulAlgo_t*>(cache.GetSubAlgo(best_algo));
  
    if (use_autotune && best_algo != nullptr) {
      best_algo = SearchBestAlgo(lt_handle,
                                 op_desc,
                                 y_desc,
                                 x_desc,
                                 out_desc,
                                 alpha,
                                 beta,
                                 y_data,
                                 x_data,
                                 out_data,
                                 ctx.stream(),
                                 workspace->ptr(),
                                 workspace_size);
    }
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmul(lt_handle,
                                                            op_desc,
                                                            alpha,
                                                            y_data,
                                                            y_desc,
                                                            x_data,
                                                            x_desc,
                                                            beta,
                                                            out_data,
                                                            out_desc,
                                                            out_data,
                                                            out_desc,
                                                            best_algo,
                                                            workspace->ptr(),
                                                            workspace_size,
                                                            ctx.stream()));
    MatmulDescCreator::Release(&op_desc, &x_desc, &y_desc, &out_desc);
  }

 private:
    static cublasLtMatmulAlgo_t* SearchBestAlgo(const phi::GPUContext& ctx,
                                                cublasLtMatmulDesc_t op_desc,
                                                cublasLtMatrixLayout_t y_desc,
                                                cublasLtMatrixLayout_t x_desc,
                                                cublasLtMatrixLayout_t out_desc,
                                                const void* alpha,
                                                const void* beta,
                                                const void* y_data,
                                                const void* x_data,
                                                void* out_data,
                                                void* workspace,
                                                size_t workspace_size) {
    const auto& stream = ctx.stream();
    cublasLtHandle_t lt_handle = ctx.cublaslt_handle();

    int best_algo_idx = -1;
    int returned_results = 0;
    constexpr int requested_algo_count = 10;
    cublasLtMatmulAlgo_t ret;
    cublasLtMatmulPreference_t preference;
    PADDLE_ENFORCE_GPU_SUCCESS(
            dynload::cublasLtMatmulPreferenceCreate(&preference));
    PADDLE_ENFORCE_GPU_SUCCESS(
            dynload::cublasLtMatmulPreferenceSetAttribute(
                preference,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace_size,
                sizeof(workspace_size)));

    std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results(
                requested_algo_count);
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulAlgoGetHeuristic(lt_handle,
                                                     op_desc,
                                                     y_desc,
                                                     x_desc,
                                                     out_desc,
                                                     out_desc,
                                                     preference,
                                                     requested_algo_count,
                                                     heuristic_results.data(),
                                                     &returned_results));
    PADDLE_ENFORCE_GT(returned_results,
                      0,
                      phi::errors::Unavailable("No GEMM algorithm avaliable."));
    
    phi::GpuTimer timer; 
    constexpr int repeats = 6;
    float min_time_cost = std::numeric_limits<float>::max();
    int best_algo = -1;
    for (int algo_idx = 0; algo_idx < returned_results; ++algo_idx) {
      ctx.Wait();
      float cur_time = 0.f;
      for (int i = 0; i < repeats; ++i) {
        timer.Start(stream);
        PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmul(lt_handle,
                                                                op_desc,
                                                                alpha,
                                                                y_data,
                                                                y_desc,
                                                                x_data,
                                                                x_desc,
                                                                beta,
                                                                out_data,
                                                                out_desc,
                                                                out_data,
                                                                out_desc,
                                                                &heuristic_results[warmup_algo_idx].algo,
                                                                workspace->ptr(),
                                                                workspace_size,
                                                                stream));
        timer.Stop(stream);
        auto time = timer.ElapsedTime();
        if (i > 0) {
          cur_time += time;
        }
      }
      VLOG(3) << "Time cost of cublaslt algo [" << algo_idx << "] is :" << (time / (repeats - 1));
      min_time_cost = (cur_time < min_time_cost) ? cur_time : min_time_cost;
      best_algo = (cur_time < min_time_cost) ? algo_idx : best_algo;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulPreferenceDestroy(preference));

    return &(heuristic_results[best_algo_idx].algo);
  }

  static phi::Allocator::AllocationPtr GetWorkspace(
                                const phi::GPUContext& ctx,
                                size_t workspace_size) {
      return paddle::memory::Alloc(
        ctx.GetPlace(),
        workspace_size,
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  }
};
#endif

}  // namespace phi


