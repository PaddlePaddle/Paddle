/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.

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
#include "paddle/fluid/platform/dynload/cublasLt.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/utils/optional.h"

DECLARE_int64(cublaslt_exhaustive_search_times);

namespace paddle {
namespace operators {

class GemmEpilogueAlgoCache {
 public:
  static GemmEpilogueAlgoCache &Instance() {
    static GemmEpilogueAlgoCache instance(
        FLAGS_cublaslt_exhaustive_search_times);
    return instance;
  }

  GemmEpilogueAlgoCache(GemmEpilogueAlgoCache const &) = delete;
  void operator=(GemmEpilogueAlgoCache const &) = delete;

  cublasLtMatmulAlgo_t *GetGemmAlgo(cublasLtHandle_t lt_handle,
                                    cublasLtMatmulDesc_t op_desc,
                                    cublasLtMatrixLayout_t a_desc,
                                    cublasLtMatrixLayout_t b_desc,
                                    cublasLtMatrixLayout_t c_desc,
                                    const void *alpha,
                                    const void *beta,
                                    const void *a,
                                    const void *b,
                                    void *c,
                                    cudaStream_t stream,
                                    void *workspace,
                                    size_t workspace_size) {
    if (search_times_ <= 0) return nullptr;

    int64_t seed = 0;
    std::hash<int64_t> hash_fn;

    HashMatmulDesc_(op_desc, &seed, hash_fn);
    HashMatrixLayoutDesc_(a_desc, &seed, hash_fn);
    HashMatrixLayoutDesc_(b_desc, &seed, hash_fn);
    HashMatrixLayoutDesc_(c_desc, &seed, hash_fn);

    cublasLtMatmulAlgo_t ret;
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      auto it = map_.find(seed);
      if (it != map_.end()) {
        return &(it->second);
      }
    }

    cublasLtMatmulPreference_t preference;
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulPreferenceCreate(&preference));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size,
            sizeof(workspace_size)));

    int returned_results = 0;
    std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results(
        requested_algo_count_);
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulAlgoGetHeuristic(
            lt_handle,
            op_desc,
            a_desc,
            b_desc,
            c_desc,
            c_desc,
            preference,
            requested_algo_count_,
            heuristic_results.data(),
            &returned_results));

    PADDLE_ENFORCE_GT(
        returned_results,
        0,
        platform::errors::Unavailable("No GEMM epilogue algorithm support!"));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulPreferenceDestroy(preference));

    int best_algo_idx = -1;
    float best_algo_time = 0;

    // Run 100 times for warmup
    int warmup_algo_idx = 0;
    for (int t = 0; t < 100; t++) {
      cublasStatus_t status = platform::dynload::cublasLtMatmul(
          lt_handle,
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
          PADDLE_THROW(platform::errors::Unavailable(
              "No GEMM epilogue algorithm support!"));
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
            platform::dynload::cublasLtMatmul(lt_handle,
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

    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(start_event));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(stop_event));

    if (best_algo_idx == -1) {
      PADDLE_THROW(
          platform::errors::Unavailable("No GEMM epilogue algorithm support!"));
    }

    ret = heuristic_results[best_algo_idx].algo;

    VLOG(4) << "Search time:" << search_times_ << ", hash-key (" << seed
            << ") not found in GemmEpilogueAlgoCache";

    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto &algo_in_map = map_[seed];
    algo_in_map = ret;
    return &algo_in_map;
  }

 private:
  explicit GemmEpilogueAlgoCache(int search_times)
      : search_times_(search_times) {
    map_.clear();
  }
  std::unordered_map<int64_t, cublasLtMatmulAlgo_t> map_;
  int search_times_;
  const int requested_algo_count_ = 10;
  std::mutex cache_mutex_;

  void HashMatmulDesc_(cublasLtMatmulDesc_t desc,
                       int64_t *seed,
                       const std::hash<int64_t> &hash_fn) {
    size_t size_to_write;
    int trans_a, trans_b;
    uint32_t epilogue;

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescGetAttribute(
            desc,
            CUBLASLT_MATMUL_DESC_TRANSA,
            &trans_a,
            sizeof(trans_a),
            &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(trans_a));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescGetAttribute(
            desc,
            CUBLASLT_MATMUL_DESC_TRANSB,
            &trans_b,
            sizeof(trans_b),
            &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(trans_b));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescGetAttribute(
            desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE,
            &epilogue,
            sizeof(epilogue),
            &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(epilogue));
  }

  void HashMatrixLayoutDesc_(cublasLtMatrixLayout_t desc,
                             int64_t *seed,
                             const std::hash<int64_t> &hash_fn) {
    size_t size_to_write;
    uint32_t dtype;
    int32_t batch;
    uint64_t row, col;
    int64_t ld, batch_offset;

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutGetAttribute(
            desc,
            CUBLASLT_MATRIX_LAYOUT_TYPE,
            &dtype,
            sizeof(dtype),
            &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(dtype));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutGetAttribute(
            desc,
            CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch,
            sizeof(batch),
            &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(batch));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutGetAttribute(
            desc,
            CUBLASLT_MATRIX_LAYOUT_ROWS,
            &row,
            sizeof(row),
            &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(row));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutGetAttribute(
            desc,
            CUBLASLT_MATRIX_LAYOUT_COLS,
            &col,
            sizeof(col),
            &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(col));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutGetAttribute(
            desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld), &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(ld));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutGetAttribute(
            desc,
            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &batch_offset,
            sizeof(batch_offset),
            &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(batch_offset));
  }

  void HashValue_(int64_t *seed,
                  const std::hash<int64_t> &hash_fn,
                  int64_t value) {
    *seed ^= hash_fn(value) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
  }
};

}  // namespace operators
}  // namespace paddle

#endif
#endif
