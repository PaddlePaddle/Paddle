/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060

#include <cuda_runtime_api.h>
#include "cuda.h"  // NOLINT
#include "paddle/phi/backends/gpu/cuda/cuda_helper.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/autotune/cache.h"
#include "paddle/phi/kernels/autotune/gpu_timer.h"

namespace phi {
namespace funcs {

enum MatmulImplType { kImplWithCublas = 1, kImplWithCublasLt = 2 };

template <typename T>
cublasComputeType_t GetCudaComputeType() {
  if (std::is_same<T, double>::value) {
    return CUBLAS_COMPUTE_64F;
  } else {
    return CUBLAS_COMPUTE_32F;
  }
}

struct MatmulDescriptor {
 public:
  cublasLtMatmulDesc_t op_desc{nullptr};
  cublasLtMatrixLayout_t x_desc{nullptr};
  cublasLtMatrixLayout_t y_desc{nullptr};
  cublasLtMatrixLayout_t out_desc{nullptr};

  template <typename T>
  void Create(const int M,
              const int N,
              const int K,
              const bool trans_x,
              const bool trans_y,
              const int batch_size = 1,
              int64_t stride_x = 0,
              int64_t stride_y = 0,
              int64_t stride_out = 0) {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;

    cudaDataType_t mat_type = phi::backends::gpu::ToCudaDataType<T>();
    cudaDataType_t scale_type = phi::backends::gpu::ToCudaDataType<MT>();
    cublasComputeType_t compute_type = GetCudaComputeType<T>();

    // Create operation desciriptor; see cublasLtMatmulDescAttributes_t for
    // details about defaults; just need to set the transforms for A and B
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type));
    cublasOperation_t cublas_trans_x = trans_x ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t cublas_trans_y = trans_y ? CUBLAS_OP_T : CUBLAS_OP_N;
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulDescSetAttribute(op_desc,
                                                CUBLASLT_MATMUL_DESC_TRANSB,
                                                &cublas_trans_x,
                                                sizeof(cublas_trans_x)));
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulDescSetAttribute(op_desc,
                                                CUBLASLT_MATMUL_DESC_TRANSA,
                                                &cublas_trans_y,
                                                sizeof(cublas_trans_y)));

    // Create matrix descriptors
    CreateMatrixLayout(&x_desc, mat_type, M, K, trans_x);
    CreateMatrixLayout(&y_desc, mat_type, K, N, trans_y);
    CreateMatrixLayout(&out_desc, mat_type, M, N, false);

    // Config batch size and stride.
    if (batch_size > 1) {
      SetBatchAndStride(x_desc, batch_size, stride_x);
      SetBatchAndStride(y_desc, batch_size, stride_y);
      SetBatchAndStride(out_desc, batch_size, stride_out);
    }
  }

  void CreateMatrixLayout(cublasLtMatrixLayout_t* desc,
                          cudaDataType type,
                          uint64_t rows,
                          uint64_t cols,
                          bool trans) {
    if (trans) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cublasLtMatrixLayoutCreate(desc, type, rows, cols, rows));
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cublasLtMatrixLayoutCreate(desc, type, cols, rows, cols));
    }
  }

  void SetBatchAndStride(cublasLtMatrixLayout_t desc,
                         int batch_size,
                         int64_t stride) {
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutSetAttribute(
        desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_size,
        sizeof(batch_size)));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutSetAttribute(
        desc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride,
        sizeof(stride)));
  }

  void Release() {
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutDestroy(y_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutDestroy(x_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutDestroy(out_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescDestroy(op_desc));

    op_desc = nullptr;
    x_desc = nullptr;
    y_desc = nullptr;
    out_desc = nullptr;
  }
};

template <typename T>
struct MatmulWithCublasLt {
 public:
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

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
    MatmulDescriptor desc;
    desc.Create<T>(M, N, K, trans_x, trans_y);
    RunImpl(ctx, desc, x_data, y_data, out_data, matmul_key);
    desc.Release();
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
    MatmulDescriptor desc;
    desc.Create<T>(
        M, N, K, trans_x, trans_y, batch_size, stride_x, stride_y, stride_out);
    RunImpl(ctx, desc, x_data, y_data, out_data, matmul_key);
    desc.Release();
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
  static phi::Allocator::AllocationPtr GetWorkspace(const phi::GPUContext& ctx,
                                                    size_t workspace_size) {
    return paddle::memory::Alloc(
        ctx.GetPlace(),
        workspace_size,
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  }

  static void RunImpl(const phi::GPUContext& ctx,
                      const MatmulDescriptor& desc,
                      const T* x_ptr,
                      const T* y_ptr,
                      T* out_ptr,
                      phi::autotune::MatmulCacheKey* matmul_key = nullptr) {
    MT alpha = static_cast<MT>(1);
    MT beta = static_cast<MT>(0);

    cublasLtHandle_t cublaslt_handle = ctx.cublaslt_handle();
    cublasLtMatmulAlgo_t* best_algo = nullptr;

    size_t workspace_size = static_cast<size_t>(4) * 1024 * 1024;
    phi::Allocator::AllocationPtr workspace = GetWorkspace(ctx, workspace_size);

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
                       cublaslt_handle,
                       desc.op_desc,
                       desc.y_desc,
                       desc.x_desc,
                       desc.out_desc,
                       static_cast<void*>(&alpha),
                       static_cast<void*>(&beta),
                       y_ptr,
                       x_ptr,
                       out_ptr,
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
        cublaslt_handle,
        desc.op_desc,
        static_cast<void*>(&alpha),
        y_ptr,
        desc.y_desc,
        x_ptr,
        desc.x_desc,
        static_cast<void*>(&beta),
        out_ptr,
        desc.out_desc,
        out_ptr,
        desc.out_desc,
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

}  // namespace funcs
}  // namespace phi

#endif
