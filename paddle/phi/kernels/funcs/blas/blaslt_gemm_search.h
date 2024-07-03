/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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

#include <glog/logging.h>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>

#include "paddle/phi/backends/dynload/cublasLt.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"

#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"

namespace phi {
namespace funcs {
namespace cublaslt_internal {

const std::array<int, 9> split_k_candidates = {2, 3, 4, 5, 6, 8, 12, 16, 32};

struct CublasLtAlgoSelectorParam {
  cublasLtMatmulAlgo_t algo;
  int m;
  int n;
  int k;
  int algo_id;
  int swizzle;
  int custom_option;
  int tile;
  int split_k_val;
  int reduction_scheme;
  int stages;
  void* workspace;
  size_t workspace_size;
  float time;
};

inline bool compare_algo_time(const CublasLtAlgoSelectorParam& param_a,
                              const CublasLtAlgoSelectorParam& param_b) {
  return (param_a.time < param_b.time);
}

class CublasLtAlgoCache {
 public:
  static CublasLtAlgoCache& Instance() {
    static CublasLtAlgoCache instance(100);
    return instance;
  }

  template <typename InT, typename OutT>
  void RunAndMeasureAlgo(cublasLtHandle_t handle,
                         cublasLtMatmulDesc_t matmul_desc,
                         cublasLtMatrixLayout_t a_desc,
                         cublasLtMatrixLayout_t b_desc,
                         cublasLtMatrixLayout_t bias_desc,
                         cublasLtMatrixLayout_t c_desc,
                         void* alpha,
                         void* beta,
                         const InT* a,
                         const InT* b,
                         const OutT* bias,
                         OutT* c,
                         CublasLtAlgoSelectorParam& param,  // NOLINT
                         cudaEvent_t& start_event,          // NOLINT
                         cudaEvent_t& stop_event,           // NOLINT
                         cudaStream_t stream) {
    cublasStatus_t status;
    cublasLtMatmulHeuristicResult_t heuristic_result;
    status = dynload::cublasLtMatmulAlgoCheck(handle,
                                              matmul_desc,
                                              a_desc,
                                              b_desc,
                                              bias_desc,
                                              c_desc,
                                              &param.algo,
                                              &heuristic_result);
    PADDLE_ENFORCE_GPU_SUCCESS(status);
    if (status != CUBLAS_STATUS_SUCCESS) {
      param.time = std::numeric_limits<float>::max();
      return;
    }

    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(start_event, stream));
    int repeats = search_times_;

    for (int loop = 0; loop < repeats; loop++) {
      status = dynload::cublasLtMatmul(handle,
                                       matmul_desc,
                                       alpha,
                                       a,
                                       a_desc,
                                       b,
                                       b_desc,
                                       beta,
                                       bias,
                                       bias_desc,
                                       c,
                                       c_desc,
                                       &param.algo,
                                       param.workspace,
                                       param.workspace_size,
                                       stream);
      if (status != CUBLAS_STATUS_SUCCESS) {
        param.time = std::numeric_limits<float>::max();
        return;
      }
    }

    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(stop_event, stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

    float time;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaEventElapsedTime(&time, start_event, stop_event));

    param.time = time / repeats;
  }

  template <typename InT, typename OutT>
  cublasLtMatmulAlgo_t* CublasLtAlgoSelect(cublasLtHandle_t handle,
                                           int m,
                                           int n,
                                           int k,
                                           int batch_count,
                                           const InT* a,
                                           const InT* b,
                                           const OutT* bias,
                                           OutT* c,
                                           void* alpha,
                                           void* beta,
                                           cublasLtMatmulDesc_t matmul_desc,
                                           cublasLtMatrixLayout_t a_desc,
                                           cublasLtMatrixLayout_t b_desc,
                                           cublasLtMatrixLayout_t bias_desc,
                                           cublasLtMatrixLayout_t c_desc,
                                           cublasComputeType_t compute_type,
                                           cudaDataType_t scale_type,
                                           cudaDataType_t a_type,
                                           cudaDataType_t b_type,
                                           cudaDataType_t bias_type,
                                           cudaDataType_t c_type,
                                           cudaStream_t stream) {
    // If we don't have config file and we donot search, here return nullptr
    if (!has_config_file_ && search_times_ <= 0) {
      return nullptr;
    }

    // VLOG(0) << "m n k: " << m << " " << n << " " << k;

    int64_t seed = 0;
    std::hash<int64_t> hash_fn;

    HashMatmulDesc(matmul_desc, &seed, hash_fn);
    HashMatrixLayoutDesc(a_desc, &seed, hash_fn);
    HashMatrixLayoutDesc(b_desc, &seed, hash_fn);
    HashMatrixLayoutDesc(bias_desc, &seed, hash_fn);
    HashMatrixLayoutDesc(c_desc, &seed, hash_fn);

    cublasLtMatmulAlgo_t ret;
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      auto it = map_.find(seed);
      if (it != map_.end()) {
        VLOG(3) << "CublasLtAlgoSelect Found in cache";
        return &(it->second);
      } else {
        // if we have cache but not found algo, and we don't want to search,
        // here return nullptr
        if (search_times_ <= 0) {
          return nullptr;
        }
      }
    }
    VLOG(3) << "CublasLtAlgoSelect Not Found in cache";

    // Get Ids
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoGetIds
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    // std::vector<int> algo_ids(requested_algo_count_);
    int algo_ids[requested_algo_count_];  // NOLINT

    int num_algo_ids;
    status = dynload::cublasLtMatmulAlgoGetIds(handle,
                                               compute_type,
                                               scale_type,
                                               a_type,
                                               b_type,
                                               bias_type,
                                               c_type,
                                               requested_algo_count_,
                                               algo_ids,
                                               &num_algo_ids);
    PADDLE_ENFORCE_GPU_SUCCESS(status);

    // Traverse all posssible algo combinations
    int step = 0;
    int limit = 20000;
    std::vector<CublasLtAlgoSelectorParam> params;

    for (int idx = 0; idx < num_algo_ids; idx++) {
      cublasLtMatmulAlgo_t algo;

      /* Initialize algo structure with given Algp ID */
      // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoInit
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulAlgoInit(handle,
                                                                 compute_type,
                                                                 scale_type,
                                                                 a_type,
                                                                 b_type,
                                                                 bias_type,
                                                                 c_type,
                                                                 algo_ids[idx],
                                                                 &algo));

      // Query the tiles enums supported by that algo which is used to alloc
      // enough space to store it
      // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoCapGetAttribute
      size_t attr_size = 0;

      int batch_support;
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulAlgoCapGetAttribute(
          &algo,
          CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT,
          &batch_support,
          sizeof(batch_support),
          &attr_size));
      if (batch_count > 1 && batch_support == 0) {
        continue;
      }

      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulAlgoCapGetAttribute(
          &algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &attr_size));

      int num_tiles = static_cast<int>(attr_size / sizeof(int));
      std::vector<int> tiles(num_tiles == 0 ? 1 : num_tiles);
      if (num_tiles == 0) {
        tiles[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
        num_tiles = 1;
      } else {
        PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulAlgoCapGetAttribute(
            &algo,
            CUBLASLT_ALGO_CAP_TILE_IDS,
            tiles.data(),
            sizeof(int) * num_tiles,
            &attr_size));
      }

      // Query the stages enums supported by that algo (cuda must >= 11.0)
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulAlgoCapGetAttribute(
          &algo, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &attr_size));
      int num_stages = static_cast<int>(attr_size / sizeof(int));
      std::vector<int> stages(num_stages == 0 ? 1 : num_stages);
      if (num_stages == 0) {
        stages[0] = CUBLASLT_MATMUL_STAGES_UNDEFINED;
        num_stages = 1;
      } else {
        PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulAlgoCapGetAttribute(
            &algo,
            CUBLASLT_ALGO_CAP_STAGES_IDS,
            stages.data(),
            sizeof(int) * num_stages,
            &attr_size));
      }

      // Retrieve Other Algo Capabilities attributes
      int splitk_support, red_mask, swizzling_max, custom_option_max;
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulAlgoCapGetAttribute(
          &algo,
          CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,
          &splitk_support,
          sizeof(splitk_support),
          &attr_size));
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulAlgoCapGetAttribute(
          &algo,
          CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK,
          &red_mask,
          sizeof(red_mask),
          &attr_size));
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulAlgoCapGetAttribute(
          &algo,
          CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,
          &swizzling_max,
          sizeof(swizzling_max),
          &attr_size));
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulAlgoCapGetAttribute(
          &algo,
          CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX,
          &custom_option_max,
          sizeof(custom_option_max),
          &attr_size));

      /* Loop over the different tiles */
      for (int tile_id = 0; tile_id < num_tiles && step < limit; tile_id++) {
        /* Loop over different stages count */
        for (int stage_id = 0; stage_id < num_stages && step < limit;
             stage_id++) {
          /* Loop over the different custom option if any */
          for (int custom_option = 0;
               custom_option <= custom_option_max && step < limit;
               custom_option++) {
            /* Loop over the CTAs swizzling support */
            for (int k = 0; k <= swizzling_max && step < limit; k++) {
              int splir_k_trial = 0;
              if (splitk_support) {
                splir_k_trial +=
                    sizeof(split_k_candidates) / sizeof(split_k_candidates[0]);
              }

              for (int l = 0; (l < (1 + splir_k_trial)) && (step < limit);
                   l++) {
                PADDLE_ENFORCE_GPU_SUCCESS(
                    dynload::cublasLtMatmulAlgoConfigSetAttribute(
                        &algo,
                        CUBLASLT_ALGO_CONFIG_TILE_ID,
                        &tiles[tile_id],
                        sizeof(tiles[tile_id])));
                PADDLE_ENFORCE_GPU_SUCCESS(
                    dynload::cublasLtMatmulAlgoConfigSetAttribute(
                        &algo,
                        CUBLASLT_ALGO_CONFIG_STAGES_ID,
                        &stages[stage_id],
                        sizeof(stages[stage_id])));
                PADDLE_ENFORCE_GPU_SUCCESS(
                    dynload::cublasLtMatmulAlgoConfigSetAttribute(
                        &algo,
                        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                        &custom_option,
                        sizeof(custom_option)));
                PADDLE_ENFORCE_GPU_SUCCESS(
                    dynload::cublasLtMatmulAlgoConfigSetAttribute(
                        &algo,
                        CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                        &k,
                        sizeof(k)));
                int split_k_val = 1;
                int reduction_scheme = CUBLASLT_REDUCTION_SCHEME_NONE;
                PADDLE_ENFORCE_GPU_SUCCESS(
                    dynload::cublasLtMatmulAlgoConfigSetAttribute(
                        &algo,
                        CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                        &split_k_val,
                        sizeof(split_k_val)));
                PADDLE_ENFORCE_GPU_SUCCESS(
                    dynload::cublasLtMatmulAlgoConfigSetAttribute(
                        &algo,
                        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                        &reduction_scheme,
                        sizeof(int)));
                if (l > 0) {  // Split-K case
                  split_k_val = split_k_candidates[l - 1];
                  PADDLE_ENFORCE_GPU_SUCCESS(
                      dynload::cublasLtMatmulAlgoConfigSetAttribute(
                          &algo,
                          CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                          &split_k_candidates[l - 1],
                          sizeof(split_k_candidates[l - 1])));
                  for (reduction_scheme = 1;
                       reduction_scheme <
                           static_cast<int>(CUBLASLT_REDUCTION_SCHEME_MASK) &&
                       (step < limit);
                       reduction_scheme = reduction_scheme << 1) {
                    if (reduction_scheme & red_mask) {
                      PADDLE_ENFORCE_GPU_SUCCESS(
                          dynload::cublasLtMatmulAlgoConfigSetAttribute(
                              &algo,
                              CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                              &reduction_scheme,
                              sizeof(reduction_scheme)));

                      cublasLtMatmulHeuristicResult_t heurResult;
                      status = dynload::cublasLtMatmulAlgoCheck(handle,
                                                                matmul_desc,
                                                                a_desc,
                                                                b_desc,
                                                                bias_desc,
                                                                c_desc,
                                                                &algo,
                                                                &heurResult);
                      if (status == CUBLAS_STATUS_SUCCESS) {
                        size_t temp_storage_bytes = heurResult.workspaceSize;
                        auto d_temp_storage = phi::memory_utils::Alloc(
                            phi::GPUPlace(
                                phi::backends::gpu::GetCurrentDeviceId()),
                            temp_storage_bytes);

                        CublasLtAlgoSelectorParam algo_select_params;
                        algo_select_params.algo = algo;
                        algo_select_params.m = m;
                        algo_select_params.n = n;
                        algo_select_params.k = k;
                        algo_select_params.algo_id = algo_ids[idx];
                        algo_select_params.tile = tiles[tile_id];
                        algo_select_params.swizzle = k;
                        algo_select_params.custom_option = custom_option;
                        algo_select_params.split_k_val = split_k_val;
                        algo_select_params.reduction_scheme = reduction_scheme;
                        algo_select_params.stages = stages[stage_id];
                        algo_select_params.workspace_size = temp_storage_bytes;
                        algo_select_params.workspace = d_temp_storage->ptr();
                        params.emplace_back(algo_select_params);
                        step++;
                      }
                    }  // end if
                  }
                } else {
                  // Prepare algos
                  cublasLtMatmulHeuristicResult_t heurResult;
                  // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoCheck
                  status = dynload::cublasLtMatmulAlgoCheck(handle,
                                                            matmul_desc,
                                                            a_desc,
                                                            b_desc,
                                                            bias_desc,
                                                            c_desc,
                                                            &algo,
                                                            &heurResult);
                  if (status == CUBLAS_STATUS_SUCCESS) {
                    size_t temp_storage_bytes = heurResult.workspaceSize;
                    auto d_temp_storage = phi::memory_utils::Alloc(
                        phi::GPUPlace(backends::gpu::GetCurrentDeviceId()),
                        temp_storage_bytes);
                    CublasLtAlgoSelectorParam algo_select_params;
                    algo_select_params.algo = algo;
                    algo_select_params.m = m;
                    algo_select_params.n = n;
                    algo_select_params.k = k;
                    algo_select_params.algo_id = algo_ids[idx];
                    algo_select_params.tile = tiles[tile_id];
                    algo_select_params.swizzle = k;
                    algo_select_params.custom_option = custom_option;
                    algo_select_params.split_k_val = split_k_val;
                    algo_select_params.reduction_scheme = reduction_scheme;
                    algo_select_params.stages = stages[stage_id];
                    algo_select_params.workspace_size = temp_storage_bytes;
                    algo_select_params.workspace = d_temp_storage->ptr();
                    params.emplace_back(algo_select_params);
                    step++;
                  }
                }
              }
            }
          }
        }
      }
    }
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&start_event));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&stop_event));

    if (step == 0) {
      VLOG(3) << "No algo can be used";
      return nullptr;
    }

    VLOG(3) << "CublasLtAlgoSelect Start testRun " << step << " "
            << params.size();

    for (int i = 0; i < step; i++) {
      RunAndMeasureAlgo(handle,
                        matmul_desc,
                        a_desc,
                        b_desc,
                        bias_desc,
                        c_desc,
                        alpha,
                        beta,
                        a,
                        b,
                        bias,
                        c,
                        params[i],
                        start_event,
                        stop_event,
                        stream);
    }
    std::sort(params.begin(), params.end(), compare_algo_time);

    size_t res_id = 0;
    while (params[res_id].time == 0) res_id++;

    if (res_id >= params.size()) {
      VLOG(3) << "No algo can be used";
      return nullptr;
    }

    VLOG(3) << "algo selected";

    ret = params[res_id].algo;
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto& algo_in_map = map_[seed];
    algo_in_map = ret;
    return &algo_in_map;
  }

  // Serialize map_ to cache file
  void serialize_algo_cache_file() {
    if (search_times_ > 0) {
      int dev;
      cudaGetDevice(&dev);
      if (dev == 0) {
        std::ofstream outfile;
        outfile.open(config_filename_, std::ios::out | std::ios::trunc);
        outfile << dynload::cublasLtGetCudartVersion() << std::endl;

        for (const auto& p : map_) {
          outfile << p.first << " ";
          for (size_t i : p.second.data) {
            outfile << i << " ";
          }
          outfile << std::endl;
        }
        outfile.close();
      }
    }
  }
  ~CublasLtAlgoCache() { serialize_algo_cache_file(); }

 private:
  explicit CublasLtAlgoCache(int search_times)
      : search_times_(search_times), has_config_file_(true) {
    // Init map_ from cache file
    std::ifstream infile;
    infile.open(config_filename_);
    if (!infile.is_open()) {
      has_config_file_ = false;
      VLOG(3) << "No CublasLtAlgoCache file found";
      return;
    }
    size_t cublaslt_version, real_cublaslt_version;
    int64_t seed = 0;
    std::array<uint64_t, 8> algo_data;
    infile >> cublaslt_version;
    VLOG(1) << "cublaslt_version " << cublaslt_version;

    if (dynload::cublasLtGetCudartVersion() != cublaslt_version) {
      LOG(INFO) << config_filename_
                << " is not compatible with current cublaslt_version "
                << real_cublaslt_version;
      return;
    }

    while (!infile.eof()) {
      infile >> seed >> algo_data[0] >> algo_data[1] >> algo_data[2] >>
          algo_data[3] >> algo_data[4] >> algo_data[5] >> algo_data[6] >>
          algo_data[7];

      for (int i = 0; i < 8; ++i) {
        map_[seed].data[i] = algo_data[i];
      }
    }
    infile.close();
  }

  std::string config_filename_{"./paddle_cublaslt_cache"};
  std::unordered_map<int64_t, cublasLtMatmulAlgo_t> map_;
  int search_times_;
  static constexpr int requested_algo_count_ = 100;
  std::mutex cache_mutex_;
  bool has_config_file_;

  inline int64_t RoundToNextHighPowOfTwo(int64_t n, int64_t min_val) {
    n--;
    n |= (n >> 1);
    n |= (n >> 2);
    n |= (n >> 4);
    n |= (n >> 8);
    n |= (n >> 16);
    return std::max(min_val, (n + 1));
  }

  void HashMatmulDesc(cublasLtMatmulDesc_t desc,
                      int64_t* seed,
                      const std::hash<int64_t>& hash_fn) {
    size_t size_to_write;
    int trans_a, trans_b;
    uint32_t epilogue;
    // int8_t fast_accum;

    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulDescGetAttribute(desc,
                                                CUBLASLT_MATMUL_DESC_TRANSA,
                                                &trans_a,
                                                sizeof(trans_a),
                                                &size_to_write));
    HashValue(seed, hash_fn, static_cast<int64_t>(trans_a));

    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulDescGetAttribute(desc,
                                                CUBLASLT_MATMUL_DESC_TRANSB,
                                                &trans_b,
                                                sizeof(trans_b),
                                                &size_to_write));
    HashValue(seed, hash_fn, static_cast<int64_t>(trans_b));

    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulDescGetAttribute(desc,
                                                CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue,
                                                sizeof(epilogue),
                                                &size_to_write));
    HashValue(seed, hash_fn, static_cast<int64_t>(epilogue));

    // PADDLE_ENFORCE_GPU_SUCCESS(
    //     dyl::cublasLtMatmulDescGetAttribute(desc,
    //                                         CUBLASLT_MATMUL_DESC_FAST_ACCUM,
    //                                         &fast_accum,
    //                                         sizeof(fast_accum),
    //                                         &size_to_write));
    // HashValue(seed, hash_fn, static_cast<int64_t>(fast_accum));
  }

  void HashMatrixLayoutDesc(cublasLtMatrixLayout_t desc,
                            int64_t* seed,
                            const std::hash<int64_t>& hash_fn) {
    size_t size_to_write;
    uint32_t dtype;
    int32_t batch;
    uint64_t row, col;
    int64_t ld, batch_offset;

    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatrixLayoutGetAttribute(desc,
                                                  CUBLASLT_MATRIX_LAYOUT_TYPE,
                                                  &dtype,
                                                  sizeof(dtype),
                                                  &size_to_write));
    HashValue(seed, hash_fn, static_cast<int64_t>(dtype));

    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutGetAttribute(
        desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch,
        sizeof(batch),
        &size_to_write));
    HashValue(seed, hash_fn, static_cast<int64_t>(batch));

    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutGetAttribute(
        desc, CUBLASLT_MATRIX_LAYOUT_ROWS, &row, sizeof(row), &size_to_write));
    HashValue(seed, hash_fn, RoundToNextHighPowOfTwo(row, 32));

    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutGetAttribute(
        desc, CUBLASLT_MATRIX_LAYOUT_COLS, &col, sizeof(col), &size_to_write));
    HashValue(seed, hash_fn, RoundToNextHighPowOfTwo(col, 32));

    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutGetAttribute(
        desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld), &size_to_write));
    HashValue(seed, hash_fn, RoundToNextHighPowOfTwo(ld, 32));

    // PADDLE_ENFORCE_GPU_SUCCESS(dyl::cublasLtMatrixLayoutGetAttribute(
    //     desc, CUBLASLT_MATRIX_LAYOUT_ROWS, &row, sizeof(row),
    //     &size_to_write));
    // HashValue(seed, hash_fn, row);

    // PADDLE_ENFORCE_GPU_SUCCESS(dyl::cublasLtMatrixLayoutGetAttribute(
    //     desc, CUBLASLT_MATRIX_LAYOUT_COLS, &col, sizeof(col),
    //     &size_to_write));
    // HashValue(seed, hash_fn, col);

    // PADDLE_ENFORCE_GPU_SUCCESS(dyl::cublasLtMatrixLayoutGetAttribute(
    //     desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld), &size_to_write));
    // HashValue(seed, hash_fn, ld);

    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutGetAttribute(
        desc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batch_offset,
        sizeof(batch_offset),
        &size_to_write));
    HashValue(seed, hash_fn, static_cast<int64_t>(batch_offset));
  }

  void HashValue(int64_t* seed,
                 const std::hash<int64_t>& hash_fn,
                 int64_t value) {
    *seed ^= hash_fn(value) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
  }
};

}  // namespace cublaslt_internal
}  // namespace funcs
}  // namespace phi
