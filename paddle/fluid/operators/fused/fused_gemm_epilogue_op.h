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

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include "paddle/fluid/platform/dynload/cublasLt.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

class GemmEpilogueAlgoCache {
 public:
  static GemmEpilogueAlgoCache &Instance() {
    static GemmEpilogueAlgoCache instance(0);
    return instance;
  }

  GemmEpilogueAlgoCache(GemmEpilogueAlgoCache const &) = delete;
  void operator=(GemmEpilogueAlgoCache const &) = delete;

  cublasLtMatmulAlgo_t GetGemmAlgo(cublasLtHandle_t lt_handle,
                                   size_t workspace_size,
                                   cublasLtMatmulDesc_t op_desc,
                                   cublasLtMatrixLayout_t a_desc,
                                   cublasLtMatrixLayout_t b_desc,
                                   cublasLtMatrixLayout_t c_desc) {
    int64_t seed = 0;
    std::hash<int64_t> hash_fn;

    HashMatmulDesc_(op_desc, &seed, hash_fn);
    HashMatrixLayoutDesc_(a_desc, &seed, hash_fn);
    HashMatrixLayoutDesc_(b_desc, &seed, hash_fn);
    HashMatrixLayoutDesc_(c_desc, &seed, hash_fn);

    cublasLtMatmulAlgo_t ret;
    auto it = map_.end();
    bool have_found = false;
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      it = map_.find(seed);

      if (it != map_.end()) {
        ret = it->second;
        have_found = true;
      }
    }

    if (!have_found) {
      cublasLtMatmulPreference_t preference;
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulPreferenceCreate(&preference));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulPreferenceSetAttribute(
              preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
              &workspace_size, sizeof(workspace_size)));

      int returned_results = 0;
      cublasLtMatmulHeuristicResult_t heuristic_results[requested_algo_count_] =
          {0};
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulAlgoGetHeuristic(
              lt_handle, op_desc, a_desc, b_desc, c_desc, c_desc, preference,
              requested_algo_count_, heuristic_results, &returned_results));

      PADDLE_ENFORCE_GE(
          returned_results, 0,
          platform::errors::NotFound("No GEMM epilogue algorithm support!"));

      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulPreferenceDestroy(preference));

      ret = heuristic_results[0].algo;

      std::lock_guard<std::mutex> lock(cache_mutex_);
      map_[seed] = ret;
    }

    return ret;
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

  void HashMatmulDesc_(cublasLtMatmulDesc_t desc, int64_t *seed,
                       const std::hash<int64_t> &hash_fn) {
    size_t size_to_write;
    int trans_a, trans_b;
    uint32_t epilogue;

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescGetAttribute(
            desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a),
            &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(trans_a));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescGetAttribute(
            desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b),
            &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(trans_b));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescGetAttribute(
            desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue),
            &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(epilogue));
  }

  void HashMatrixLayoutDesc_(cublasLtMatrixLayout_t desc, int64_t *seed,
                             const std::hash<int64_t> &hash_fn) {
    size_t size_to_write;
    uint32_t dtype;
    int32_t batch;
    uint64_t row, col;
    int64_t ld, batch_offset;

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutGetAttribute(
            desc, CUBLASLT_MATRIX_LAYOUT_TYPE, &dtype, sizeof(dtype),
            &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(dtype));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutGetAttribute(
            desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch),
            &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(batch));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutGetAttribute(
            desc, CUBLASLT_MATRIX_LAYOUT_ROWS, &row, sizeof(row),
            &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(row));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutGetAttribute(
            desc, CUBLASLT_MATRIX_LAYOUT_COLS, &col, sizeof(col),
            &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(col));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutGetAttribute(
            desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld), &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(ld));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutGetAttribute(
            desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_offset,
            sizeof(batch_offset), &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(batch_offset));
  }

  void HashValue_(int64_t *seed, const std::hash<int64_t> &hash_fn,
                  int64_t value) {
    *seed ^= hash_fn(value) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
  }
};

}  // namespace operators
}  // namespace paddle
