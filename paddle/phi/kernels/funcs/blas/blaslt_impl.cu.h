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
#include "paddle/phi/kernels/autotune/cache.h"
#include "paddle/phi/kernels/autotune/gpu_timer.h"
#endif

namespace phi {

#if CUDA_VERSION >= 11060
enum MatmulImplType { kImplWithCublas = 1, kImplWithCublasLt = 2 };

template <typename T>
struct GetCublasMatType {
  static cudaDataType_t Get() { return CUDA_R_32F; }
};

template <>
struct GetCublasMatType<phi::dtype::float16> {
  static cudaDataType_t Get() { return CUDA_R_16F; }
};

template <>
struct GetCublasMatType<phi::dtype::bfloat16> {
  static cudaDataType_t Get() { return CUDA_R_16BF; }
};

template <>
struct GetCublasMatType<double> {
  static cudaDataType_t Get() { return CUDA_R_64F; }
};

template <typename T>
struct AlphaBetaTraits {
 public:
  using Type = float;
};

template <>
struct AlphaBetaTraits<double> {
 public:
  using Type = double;
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
                     const cublasOperation_t& trans_x,
                     const cublasOperation_t& trans_y,
                     const int batch_size = 1,
                     int64_t stride_x = 0,
                     int64_t stride_y = 0,
                     int64_t stride_out = 0) {
    // Create operation desciriptor; see cublasLtMatmulDescAttributes_t for
    // details about defaults; just need to set the transforms for A and B
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulDescCreate(op_desc, compute_type, scale_type));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescSetAttribute(
        *op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_x, sizeof(trans_x)));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescSetAttribute(
        *op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_y, sizeof(trans_y)));

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
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cublasLtMatrixLayoutCreate(y_desc, mat_type, N, K, N));
    }
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatrixLayoutCreate(out_desc, mat_type, N, M, N));

    // Config batch size and stride.
    if (batch_size > 1) {
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
  }

  static phi::Allocator::AllocationPtr GetWorkspace(const phi::GPUContext& ctx,
                                                    size_t workspace_size) {
    return paddle::memory::Alloc(
        ctx.GetPlace(),
        workspace_size,
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  }

  static void Release(cublasLtMatmulDesc_t* op_desc,
                      cublasLtMatrixLayout_t* x_desc,
                      cublasLtMatrixLayout_t* y_desc,
                      cublasLtMatrixLayout_t* out_desc) {
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutDestroy(*y_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutDestroy(*x_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutDestroy(*out_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescDestroy(*op_desc));
  }
};

template <typename Context, typename T>
struct MatmulWithCublasLt {
  static void Run(const Context& ctx,
                  const T* x_data,
                  const T* y_data,
                  T* out_data,
                  const int M,
                  const int N,
                  const int K,
                  const bool trans_x,
                  const bool trans_y,
                  phi::autotune::MatmulCacheKey* matmul_key = nullptr) {}

  static void RunWithBatch(
      const Context& ctx,
      const T* x_data,
      const T* y_data,
      T* out_data,
      const int M,
      const int N,
      const int K,
      bool trans_x,
      bool trans_y,
      int batch_size,
      int64_t stride_x,
      int64_t stride_y,
      int64_t stride_out,
      phi::autotune::MatmulCacheKey* matmul_key = nullptr) {}

  static void RunWithBatch(
      const Context& ctx,
      const T** x_data,
      const T** y_data,
      T** out_data,
      const int M,
      const int N,
      const int K,
      bool trans_x,
      bool trans_y,
      int batch_size,
      phi::autotune::MatmulCacheKey* matmul_key = nullptr) {}
};

template <typename T>
struct MatmulWithCublasLt<phi::GPUContext, T> {
 public:
  static void Run(const phi::GPUContext& ctx,
                  const T* x_data,
                  const T* y_data,
                  T* out_data,
                  const int M,
                  const int N,
                  const int K,
                  const bool trans_x,
                  const bool trans_y,
                  phi::autotune::MatmulCacheKey* matmul_key = nullptr) {
    using MT = typename AlphaBetaTraits<T>::Type;

    // Init data structure
    cublasLtMatmulDesc_t op_desc = NULL;
    cublasLtMatrixLayout_t x_desc = NULL, y_desc = NULL, out_desc = NULL;
    cublasOperation_t transx = trans_x ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transy = trans_y ? CUBLAS_OP_T : CUBLAS_OP_N;

    cudaDataType_t mat_type = GetCublasMatType<T>::Get();
    cudaDataType_t scale_type = CUDA_R_32F;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    if (std::is_same<T, double>::value) {
      scale_type = CUDA_R_64F;
      compute_type = CUBLAS_COMPUTE_64F;
    }

    MT alpha_data = static_cast<MT>(0);
    MT beta_data = static_cast<MT>(0);
    void* alpha = &alpha_data;
    void* beta = &beta_data;
    MatmulDescCreator::Create(&op_desc,
                              &x_desc,
                              &y_desc,
                              &out_desc,
                              compute_type,
                              mat_type,
                              scale_type,
                              M,
                              N,
                              K,
                              transx,
                              transy);
    RunImpl(ctx,
            op_desc,
            x_desc,
            y_desc,
            out_desc,
            x_data,
            y_data,
            out_data,
            alpha,
            beta,
            matmul_key);
    MatmulDescCreator::Release(&op_desc, &x_desc, &y_desc, &out_desc);
  }

  static void RunWithBatch(
      const phi::GPUContext& ctx,
      const T* x_data,
      const T* y_data,
      T* out_data,
      const int M,
      const int N,
      const int K,
      bool trans_x,
      bool trans_y,
      int batch_size,
      int64_t stride_x,
      int64_t stride_y,
      int64_t stride_out,
      phi::autotune::MatmulCacheKey* matmul_key = nullptr) {
    using MT = typename AlphaBetaTraits<T>::Type;

    cublasLtMatmulDesc_t op_desc = NULL;
    cublasLtMatrixLayout_t x_desc = NULL, y_desc = NULL, out_desc = NULL;
    cublasOperation_t transx = trans_x ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transy = trans_y ? CUBLAS_OP_T : CUBLAS_OP_N;

    cudaDataType_t mat_type = GetCublasMatType<T>::Get();
    cudaDataType_t scale_type = CUDA_R_32F;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    if (std::is_same<T, double>::value) {
      scale_type = CUDA_R_64F;
      compute_type = CUBLAS_COMPUTE_64F;
    }

    MT alpha_data = static_cast<MT>(0);
    MT beta_data = static_cast<MT>(0);
    MatmulDescCreator::Create(&op_desc,
                              &x_desc,
                              &y_desc,
                              &out_desc,
                              compute_type,
                              mat_type,
                              scale_type,
                              M,
                              N,
                              K,
                              transx,
                              transy,
                              batch_size,
                              stride_x,
                              stride_y,
                              stride_out);
    RunImpl(ctx,
            op_desc,
            x_desc,
            y_desc,
            out_desc,
            x_data,
            y_data,
            out_data,
            static_cast<void*>(&alpha_data),
            static_cast<void*>(&beta_data),
            matmul_key);
    MatmulDescCreator::Release(&op_desc, &x_desc, &y_desc, &out_desc);
  }

  static void RunWithBatch(
      const phi::GPUContext& ctx,
      const T** x_data,
      const T** y_data,
      T** out_data,
      const int M,
      const int N,
      const int K,
      bool trans_x,
      bool trans_y,
      int batch_size,
      phi::autotune::MatmulCacheKey* matmul_key = nullptr) {
    for (int i = 0; i < batch_size; ++i) {
      Run(ctx,
          x_data[i],
          y_data[i],
          out_data[i],
          M,
          N,
          K,
          trans_x,
          trans_y,
          matmul_key);
    }
  }

 private:
  static void RunImpl(const phi::GPUContext& ctx,
                      const cublasLtMatmulDesc_t& op_desc,
                      const cublasLtMatrixLayout_t& x_desc,
                      const cublasLtMatrixLayout_t& y_desc,
                      const cublasLtMatrixLayout_t& out_desc,
                      const T* x_data,
                      const T* y_data,
                      T* out_data,
                      void* alpha,
                      void* beta,
                      phi::autotune::MatmulCacheKey* matmul_key = nullptr) {
    cublasLtHandle_t lt_handle = ctx.cublaslt_handle();
    cublasLtMatmulAlgo_t* best_algo = nullptr;

    size_t workspace_size = static_cast<size_t>(4) * 1024 * 1024;
    phi::Allocator::AllocationPtr workspace =
        MatmulDescCreator::GetWorkspace(ctx, workspace_size);

    if (matmul_key != nullptr) {
      auto& cache = phi::autotune::AutoTuneCache::Instance().GetMatmul();
      size_t sub_key = matmul_key->GetSubKey(
          static_cast<int64_t>(MatmulImplType::kImplWithCublasLt));
      if (cache.FindSubKey(sub_key)) {
        best_algo =
            reinterpret_cast<cublasLtMatmulAlgo_t*>(cache.GetSubKey(sub_key));
      } else if (phi::autotune::AutoTuneStatus::Instance().UseAutoTune()) {
        cublasLtMatmulAlgo_t test_algo;
        SearchBestAlgo(ctx,
                       lt_handle,
                       op_desc,
                       y_desc,
                       x_desc,
                       out_desc,
                       alpha,
                       beta,
                       y_data,
                       x_data,
                       out_data,
                       workspace->ptr(),
                       workspace_size,
                       &(test_algo));
        cache.SetSubKey(
            sub_key,
            reinterpret_cast<phi::autotune::MatmulHashValueType*>(&test_algo));
        best_algo = &test_algo;
      }
    }

    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmul(
        lt_handle,
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
        reinterpret_cast<cublasLtMatmulAlgo_t*>(best_algo),
        workspace->ptr(),
        workspace_size,
        ctx.stream()));
  }

  static void SearchBestAlgo(const phi::GPUContext& ctx,
                             const cublasLtHandle_t& lt_handle,
                             const cublasLtMatmulDesc_t& op_desc,
                             const cublasLtMatrixLayout_t& y_desc,
                             const cublasLtMatrixLayout_t& x_desc,
                             const cublasLtMatrixLayout_t& out_desc,
                             const void* alpha,
                             const void* beta,
                             const void* y_data,
                             const void* x_data,
                             void* out_data,
                             void* workspace_ptr,
                             size_t workspace_size,
                             cublasLtMatmulAlgo_t* best_algo) {
    const auto& stream = ctx.stream();
    int returned_results = 0;
    constexpr int requested_algo_count = 10;
    cublasLtMatmulPreference_t preference;

    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulPreferenceCreate(&preference));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulPreferenceSetAttribute(
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
    int best_algo_idx = -1;
    constexpr int repeats = 6;
    float min_time_cost = std::numeric_limits<float>::max();
    for (int algo_idx = 0; algo_idx < returned_results; ++algo_idx) {
      ctx.Wait();
      float cur_time = 0.f;
      for (int i = 0; i < repeats; ++i) {
        timer.Start(stream);
        PADDLE_ENFORCE_GPU_SUCCESS(
            dynload::cublasLtMatmul(lt_handle,
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
                                    &(heuristic_results[algo_idx].algo),
                                    workspace_ptr,
                                    workspace_size,
                                    stream));
        timer.Stop(stream);
        auto time = timer.ElapsedTime();
        if (i > 0) {
          cur_time += time;
        }
      }
      float time_cnt = (cur_time / (repeats - 1));
      VLOG(4) << "Time cost in MatmulWithCublaslt algo[" << algo_idx << "]"
              << "is : " << time_cnt << " s";

      if (cur_time < min_time_cost) {
        best_algo_idx = algo_idx;
        min_time_cost = cur_time;
      }
    }
    VLOG(4) << "Best_algo_idx in MatmulWithCublaslt is : " << best_algo_idx;

    *best_algo = heuristic_results[best_algo_idx].algo;
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulPreferenceDestroy(preference));
  }
};
#endif  // CUDA_VERSION >= 11060

}  // namespace phi
