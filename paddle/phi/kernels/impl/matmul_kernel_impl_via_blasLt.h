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

#include <algorithm>
#include <mutex>
#include <unordered_map>

#include "gflags/gflags.h"
#include "paddle/phi/backends/dynload/cublasLt.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/utils/optional.h"

DECLARE_int64(cublaslt_exhaustive_search_times);

namespace phi {

class MatmulAlgoCache {
 public:
  static MatmulAlgoCache& Instance() {
    static MatmulAlgoCache instance(FLAGS_cublaslt_exhaustive_search_times);
    return instance;
  }

  MatmulAlgoCache(MatmulAlgoCache const&) = delete;
  void operator=(MatmulAlgoCache const&) = delete;

  cublasLtMatmulAlgo_t* GetGemmAlgo(cublasLtHandle_t lt_handle,
                                    cublasLtMatmulDesc_t op_desc,
                                    cublasLtMatrixLayout_t a_desc,
                                    cublasLtMatrixLayout_t b_desc,
                                    cublasLtMatrixLayout_t c_desc,
                                    const void* alpha,
                                    const void* beta,
                                    const void* a,
                                    const void* b,
                                    void* c,
                                    cudaStream_t stream,
                                    void* workspace,
                                    size_t workspace_size) {
    if (search_times_ <= 0) return nullptr;

    int64_t seed = 0;
    std::hash<int64_t> hash_fn;

    HashMatmulDesc_(op_desc, &seed, hash_fn);
    HashMatrixLayoutDesc_(a_desc, &seed, hash_fn);
    HashMatrixLayoutDesc_(b_desc, &seed, hash_fn);
    HashMatrixLayoutDesc_(c_desc, &seed, hash_fn);

    cublasLtMatmulAlgo_t ret;
    // FIXME: this code block:
    // std::lock_guard will call cache_mutex_.unlock() when it is destructed.
    // The purpose of the braces is to precisely control when it is destructed.
    // but why does it affect the Nsys of the cublasLtMatmul call.
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      auto it = map_.find(seed);
      if (it != map_.end()) {
        return &(it->second);
      }
    }
    cublasLtMatmulPreference_t preference;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cublasLtMatmulPreferenceCreate(&preference));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size,
            sizeof(workspace_size)));

    int returned_results = 0;
    std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results(
        requested_algo_count_);
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cublasLtMatmulAlgoGetHeuristic(lt_handle,
                                                     op_desc,
                                                     a_desc,
                                                     b_desc,
                                                     c_desc,
                                                     c_desc,
                                                     preference,
                                                     requested_algo_count_,
                                                     heuristic_results.data(),
                                                     &returned_results));

    PADDLE_ENFORCE_GT(returned_results,
                      0,
                      phi::errors::Unavailable("No GEMM algorithm avaliable."));

    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cublasLtMatmulPreferenceDestroy(preference));

    int best_algo_idx = -1;
    float best_algo_time = 0;

    // For warmup
    int warmup_algo_idx = 0;
    for (int t = 0; t < 100; t++) {
      cublasStatus_t status =
          phi::dynload::cublasLtMatmul(lt_handle,
                                       op_desc,
                                       alpha,
                                       a,
                                       a_desc,
                                       b,
                                       b_desc,
                                       beta,
                                       c,
                                       c_desc,
                                       c,
                                       c_desc,
                                       &heuristic_results[warmup_algo_idx].algo,
                                       workspace,
                                       workspace_size,
                                       stream);
      if (status != CUBLAS_STATUS_SUCCESS) {
        t = -1;
        warmup_algo_idx += 1;
        if (warmup_algo_idx == requested_algo_count_) {
          PADDLE_THROW(
              phi::errors::Unavailable("No GEMM algorithm avaliable."));
        }
      }
    }

    cudaEvent_t start_event, stop_event;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&start_event));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&stop_event));

    for (int algo_idx = 0; algo_idx < returned_results; ++algo_idx) {
      float curr_time = 0;
      for (int check_idx = 0; check_idx < search_times_; check_idx++) {
        float time = 0;
        PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(start_event, stream));

        cublasStatus_t status =
            phi::dynload::cublasLtMatmul(lt_handle,
                                         op_desc,
                                         alpha,
                                         a,
                                         a_desc,
                                         b,
                                         b_desc,
                                         beta,
                                         c,
                                         c_desc,
                                         c,
                                         c_desc,
                                         &heuristic_results[algo_idx].algo,
                                         workspace,
                                         workspace_size,
                                         stream);

        PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(stop_event, stream));
        PADDLE_ENFORCE_GPU_SUCCESS(cudaEventSynchronize(stop_event));
        PADDLE_ENFORCE_GPU_SUCCESS(
            cudaEventElapsedTime(&time, start_event, stop_event));
        curr_time += time;
        if (status != CUBLAS_STATUS_SUCCESS) {
          curr_time = 3.40282e+038;  // Max Value of float
          break;
        }
      }

      curr_time = curr_time / search_times_;
      if (curr_time < best_algo_time || algo_idx == 0) {
        best_algo_idx = algo_idx;
        best_algo_time = curr_time;
      }
    }

    // std::cout <<  "Found Best Algo : " << best_algo_idx << std::endl;
    // std::cout <<  "best_algo_time : " << best_algo_time << std::endl;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(start_event));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(stop_event));

    if (best_algo_idx == -1) {
      PADDLE_THROW(phi::errors::Unavailable("No GEMM algorithm avaliable."));
    }

    ret = heuristic_results[best_algo_idx].algo;

    VLOG(4) << "Search time:" << search_times_ << ", hash-key (" << seed
            << ") not found in MatmulAlgoCache";

    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto& algo_in_map = map_[seed];
    algo_in_map = ret;
    return &algo_in_map;
  }

 private:
  explicit MatmulAlgoCache(int search_times) : search_times_(search_times) {
    map_.clear();
  }
  std::unordered_map<int64_t, cublasLtMatmulAlgo_t> map_;
  int search_times_;
  const int requested_algo_count_ = 10;
  std::mutex cache_mutex_;

  void HashMatmulDesc_(cublasLtMatmulDesc_t desc,
                       int64_t* seed,
                       const std::hash<int64_t>& hash_fn) {
    size_t size_to_write;
    int trans_a, trans_b;
    uint32_t epilogue;

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmulDescGetAttribute(
        desc,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &trans_a,
        sizeof(trans_a),
        &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(trans_a));

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmulDescGetAttribute(
        desc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &trans_b,
        sizeof(trans_b),
        &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(trans_b));

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmulDescGetAttribute(
        desc,
        CUBLASLT_MATMUL_DESC_EPILOGUE,
        &epilogue,
        sizeof(epilogue),
        &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(epilogue));
  }

  void HashMatrixLayoutDesc_(cublasLtMatrixLayout_t desc,
                             int64_t* seed,
                             const std::hash<int64_t>& hash_fn) {
    size_t size_to_write;
    uint32_t dtype;
    int32_t batch;
    uint64_t row, col;
    int64_t ld, batch_offset;

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatrixLayoutGetAttribute(
        desc,
        CUBLASLT_MATRIX_LAYOUT_TYPE,
        &dtype,
        sizeof(dtype),
        &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(dtype));

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatrixLayoutGetAttribute(
        desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch,
        sizeof(batch),
        &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(batch));

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatrixLayoutGetAttribute(
        desc, CUBLASLT_MATRIX_LAYOUT_ROWS, &row, sizeof(row), &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(row));

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatrixLayoutGetAttribute(
        desc, CUBLASLT_MATRIX_LAYOUT_COLS, &col, sizeof(col), &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(col));

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatrixLayoutGetAttribute(
        desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld), &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(ld));

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatrixLayoutGetAttribute(
        desc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batch_offset,
        sizeof(batch_offset),
        &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(batch_offset));
  }

  void HashValue_(int64_t* seed,
                  const std::hash<int64_t>& hash_fn,
                  int64_t value) {
    *seed ^= hash_fn(value) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
  }
};

template <typename T, class Context>
struct CublasLtGEMM {
  void operator()(const Context& dev_ctx,
                  const T* x_data,
                  const T* y_data,
                  const int M,
                  const int N,
                  const int K,
                  T* out_data,
                  bool trans_x,
                  bool trans_y,
                  bool* isCublasLt) {}
};

template <typename T, class Context>
struct CublasLtBatchedGEMM {
  void operator()(const Context& dev_ctx,
                  const T* x_data,
                  const T* y_data,
                  const int M,
                  const int N,
                  const int K,
                  T* out_data,
                  bool trans_x,
                  bool trans_y,
                  int batch_size,
                  int64_t stride_x,
                  int64_t stride_y,
                  int64_t stride_out,
                  bool* isCublasLt) {}
  void operator()(const Context& dev_ctx,
                  const T** x_data,
                  const T** y_data,
                  const int M,
                  const int N,
                  const int K,
                  T** out_data,
                  bool trans_x,
                  bool trans_y,
                  int batch_size,
                  bool* isCublasLt) {}
};

template <typename T>
struct CublasLtGEMM<T, phi::GPUContext> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const T* x_data,
                  const T* y_data,
                  const int M,
                  const int N,
                  const int K,
                  T* out_data,
                  bool trans_x,
                  bool trans_y,
                  bool* isCublasLt) {
    // init data structure
    cublasLtHandle_t lt_handle = dev_ctx.cublaslt_handle();

    cublasLtMatmulDesc_t operation_desc = NULL;
    cublasLtMatrixLayout_t x_desc = NULL, y_desc = NULL, out_desc = NULL;

    cublasOperation_t transx = trans_x ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transy = trans_y ? CUBLAS_OP_T : CUBLAS_OP_N;

    cudaDataType_t mat_type = CUDA_R_32F;
    cudaDataType_t scale_type = CUDA_R_32F;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

    if (std::is_same<T, phi::dtype::float16>::value) {
      mat_type = CUDA_R_16F;
    }
    if (std::is_same<T, phi::dtype::bfloat16>::value) {
      mat_type = CUDA_R_16BF;
    }
    if (std::is_same<T, double>::value) {
      mat_type = CUDA_R_64F;
      scale_type = CUDA_R_64F;
      compute_type = CUBLAS_COMPUTE_64F;
    }

    // Create operation desciriptor; see cublasLtMatmulDescAttributes_t for
    // details about defaults; This OP we just need to set the transforms for A
    // and B
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmulDescCreate(
        &operation_desc, compute_type, scale_type));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmulDescSetAttribute(
        operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transx, sizeof(transx)));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmulDescSetAttribute(
        operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transy, sizeof(transy)));

    // Create matrix descriptors
    if (trans_x)
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasLtMatrixLayoutCreate(&x_desc, mat_type, M, K, M));
    else
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasLtMatrixLayoutCreate(&x_desc, mat_type, K, M, K));
    if (trans_y)
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasLtMatrixLayoutCreate(&y_desc, mat_type, K, N, K));
    else
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasLtMatrixLayoutCreate(&y_desc, mat_type, N, K, N));

    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cublasLtMatrixLayoutCreate(&out_desc, mat_type, N, M, N));

    double alpha64 = 1.0, beta64 = 0.0;
    float alpha32 = 1.0f, beta32 = 0.0f;
    void *alpha = nullptr, *beta = nullptr;
    if (std::is_same<T, double>::value) {
      alpha = &alpha64;
      alpha = &beta64;
    } else {
      alpha = &alpha32;
      beta = &beta32;
    }

    size_t workspace_size = static_cast<size_t>(4) * 1024 * 1024;
    cudaStream_t stream = dev_ctx.stream();
    phi::Allocator::AllocationPtr workspace = paddle::memory::Alloc(
        dev_ctx.GetPlace(),
        workspace_size,
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

    auto algo = MatmulAlgoCache::Instance().GetGemmAlgo(lt_handle,
                                                        operation_desc,
                                                        y_desc,
                                                        x_desc,
                                                        out_desc,
                                                        alpha,
                                                        beta,
                                                        y_data,
                                                        x_data,
                                                        out_data,
                                                        stream,
                                                        workspace->ptr(),
                                                        workspace_size);
    // We can take the advantage of cublasLtMatmul shortcut notation with
    // algo = NULL which will force matmul to get the basic heuristic result
    // internally. Downsides of this approach are that there is no way to
    // configure search preferences (e.g. disallow tensor operations or some
    // reduction schemes) and no way to store the algo for later use
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmul(lt_handle,
                                                            operation_desc,
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
                                                            algo,
                                                            workspace->ptr(),
                                                            workspace_size,
                                                            stream));
    // Descriptors are no longer needed as all GPU work was already enqueued
    if (y_desc)
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasLtMatrixLayoutDestroy(y_desc));
    if (x_desc)
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasLtMatrixLayoutDestroy(x_desc));
    if (out_desc)
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasLtMatrixLayoutDestroy(out_desc));
    if (operation_desc)
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasLtMatmulDescDestroy(operation_desc));
    *isCublasLt = true;
    return;
  }
};

template <typename T>
struct CublasLtBatchedGEMM<T, phi::GPUContext> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const T* x_data,
                  const T* y_data,
                  const int M,
                  const int N,
                  const int K,
                  T* out_data,
                  bool trans_x,
                  bool trans_y,
                  int batch_size,
                  int64_t stride_x,
                  int64_t stride_y,
                  int64_t stride_out,
                  bool* isCublasLt) {
    // init data structure
    cublasLtHandle_t lt_handle = dev_ctx.cublaslt_handle();

    cublasLtMatmulDesc_t operation_desc = NULL;
    cublasLtMatrixLayout_t x_desc = NULL, y_desc = NULL, out_desc = NULL;

    cublasOperation_t transx = trans_x ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transy = trans_y ? CUBLAS_OP_T : CUBLAS_OP_N;

    cudaDataType_t mat_type = CUDA_R_32F;
    cudaDataType_t scale_type = CUDA_R_32F;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

    if (std::is_same<T, phi::dtype::float16>::value) {
      mat_type = CUDA_R_16F;
    }
    if (std::is_same<T, phi::dtype::bfloat16>::value) {
      mat_type = CUDA_R_16BF;
    }
    if (std::is_same<T, double>::value) {
      mat_type = CUDA_R_64F;
      scale_type = CUDA_R_64F;
      compute_type = CUBLAS_COMPUTE_64F;
    }

    // Create operation desciriptor; see cublasLtMatmulDescAttributes_t for
    // details about defaults; This OP we just need to set the transforms for A
    // and B
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmulDescCreate(
        &operation_desc, compute_type, scale_type));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmulDescSetAttribute(
        operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transx, sizeof(transx)));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmulDescSetAttribute(
        operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transy, sizeof(transy)));

    // Create matrix descriptors, this case batch size and counts should be
    // configured
    if (trans_x)
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasLtMatrixLayoutCreate(&x_desc, mat_type, M, K, M));
    else
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasLtMatrixLayoutCreate(&x_desc, mat_type, K, M, K));
    if (trans_y)
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasLtMatrixLayoutCreate(&y_desc, mat_type, K, N, K));
    else
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasLtMatrixLayoutCreate(&y_desc, mat_type, N, K, N));

    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cublasLtMatrixLayoutCreate(&out_desc, mat_type, N, M, N));

    // Config batch size and counts
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatrixLayoutSetAttribute(
        x_desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_size,
        sizeof(batch_size)));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatrixLayoutSetAttribute(
        x_desc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_x,
        sizeof(stride_x)));

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatrixLayoutSetAttribute(
        y_desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_size,
        sizeof(batch_size)));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatrixLayoutSetAttribute(
        y_desc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_y,
        sizeof(stride_y)));

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatrixLayoutSetAttribute(
        out_desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_size,
        sizeof(batch_size)));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatrixLayoutSetAttribute(
        out_desc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_out,
        sizeof(stride_out)));

    double alpha64 = 1.0, beta64 = 0.0;
    float alpha32 = 1.0f, beta32 = 0.0f;
    void *alpha = nullptr, *beta = nullptr;
    if (std::is_same<T, double>::value) {
      alpha = &alpha64;
      alpha = &beta64;
    } else {
      alpha = &alpha32;
      beta = &beta32;
    }

    size_t workspace_size = static_cast<size_t>(4) * 1024 * 1024;
    cudaStream_t stream = dev_ctx.stream();
    phi::Allocator::AllocationPtr workspace = paddle::memory::Alloc(
        dev_ctx.GetPlace(),
        workspace_size,
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

    auto algo = MatmulAlgoCache::Instance().GetGemmAlgo(lt_handle,  // OK
                                                        operation_desc,
                                                        y_desc,
                                                        x_desc,
                                                        out_desc,
                                                        alpha,
                                                        beta,
                                                        y_data,
                                                        x_data,
                                                        out_data,
                                                        stream,
                                                        workspace->ptr(),
                                                        workspace_size);
    // We can take the advantage of cublasLtMatmul shortcut notation with
    // algo = NULL which will force matmul to get the basic heuristic result
    // internally. Downsides of this approach are that there is no way to
    // configure search preferences (e.g. disallow tensor operations or some
    // reduction schemes) and no way to store the algo for later use
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmul(lt_handle,
                                                            operation_desc,
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
                                                            algo,
                                                            workspace->ptr(),
                                                            workspace_size,
                                                            stream));
    // Descriptors are no longer needed as all GPU work was already enqueued
    if (y_desc)
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasLtMatrixLayoutDestroy(y_desc));
    if (x_desc)
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasLtMatrixLayoutDestroy(x_desc));
    if (out_desc)
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasLtMatrixLayoutDestroy(out_desc));
    if (operation_desc)
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasLtMatmulDescDestroy(operation_desc));
    *isCublasLt = true;
    return;
  }
  void operator()(const phi::GPUContext& dev_ctx,
                  const T** x_data,
                  const T** y_data,
                  const int M,
                  const int N,
                  const int K,
                  T** out_data,
                  bool trans_x,
                  bool trans_y,
                  int batch_size,
                  bool* isCublasLt) {
    for (int k = 0; k < batch_size; ++k) {
      CublasLtGEMM<T, phi::GPUContext>()(dev_ctx,
                                         x_data[k],
                                         y_data[k],
                                         M,
                                         N,
                                         K,
                                         out_data[k],
                                         trans_x,
                                         trans_y,
                                         isCublasLt);
    }
  }
};

}  // namespace phi

#endif
#endif
