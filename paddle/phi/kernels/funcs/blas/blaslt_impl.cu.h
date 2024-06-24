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

#include "glog/logging.h"

#include <cuda_runtime_api.h>  // NOLINT
#include "cuda.h"              // NOLINT
#include "paddle/phi/backends/dynload/cublasLt.h"
#include "paddle/phi/backends/gpu/cuda/cuda_helper.h"

#include "paddle/common/flags.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/autotune/gpu_timer.h"
#include "paddle/phi/kernels/autotune/switch_autotune.h"

COMMON_DECLARE_int64(cublaslt_exhaustive_search_times);
COMMON_DECLARE_bool(enable_blaslt_global_search);
#endif

namespace phi {
namespace funcs {
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060
namespace cutlass_internal {

#define PADDLE_CUBLASLT_STATUS_CHECK(name)                                    \
  PADDLE_ENFORCE_EQ(                                                          \
      status,                                                                 \
      CUBLAS_STATUS_SUCCESS,                                                  \
      phi::errors::External(                                                  \
          #name                                                               \
          "execution error"                                                   \
          "refer https://docs.nvidia.com/cuda/cublas/index.html to get more " \
          "information"))

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

#if CUDA_VERSION >= 11020
class CublasLtAlgoCache {
 public:
  static CublasLtAlgoCache& Instance() {
    static CublasLtAlgoCache instance(100);
    return instance;
  }

  template <typename InT, typename OutT>
  void TestMatmulRun(cublasLtHandle_t handle,
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
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoCheck);
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

    // VLOG(0) << "m n k" << m << " " << n << " " << k;

    int64_t seed = 0;
    std::hash<int64_t> hash_fn;

    HashMatmulDesc_(matmul_desc, &seed, hash_fn);
    HashMatrixLayoutDesc_(a_desc, &seed, hash_fn);
    HashMatrixLayoutDesc_(b_desc, &seed, hash_fn);
    HashMatrixLayoutDesc_(bias_desc, &seed, hash_fn);
    HashMatrixLayoutDesc_(c_desc, &seed, hash_fn);

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
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoGetIds);

    // Traverse all posssible algo combinations
    int step = 0;
    int limit = 20000;
    std::vector<CublasLtAlgoSelectorParam> params;

    VLOG(3) << "cublasLtMatmulAlgoGetIds: " << num_algo_ids;

    for (int idx = 0; idx < num_algo_ids; idx++) {
      cublasLtMatmulAlgo_t algo;

      /* Initialize algo structure with given Algp ID */
      // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoInit
      status = dynload::cublasLtMatmulAlgoInit(handle,
                                               compute_type,
                                               scale_type,
                                               a_type,
                                               b_type,
                                               bias_type,
                                               c_type,
                                               algo_ids[idx],
                                               &algo);
      PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoInit);

      // Query the tiles enums supported by that algo which is used to alloc
      // enough space to store it
      // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoCapGetAttribute
      size_t attr_size = 0;

      int batch_support;
      status = dynload::cublasLtMatmulAlgoCapGetAttribute(
          &algo,
          CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT,
          &batch_support,
          sizeof(batch_support),
          &attr_size);
      PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoCapGetAttribute);
      if (batch_count > 1 && batch_support == 0) {
        continue;
      }

      status = dynload::cublasLtMatmulAlgoCapGetAttribute(
          &algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &attr_size);
      PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoCapGetAttribute);

      int num_tiles = static_cast<int>(attr_size / sizeof(int));
      std::vector<int> tiles(num_tiles == 0 ? 1 : num_tiles);
      if (num_tiles == 0) {
        tiles[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
        num_tiles = 1;
      } else {
        status = dynload::cublasLtMatmulAlgoCapGetAttribute(
            &algo,
            CUBLASLT_ALGO_CAP_TILE_IDS,
            tiles.data(),
            sizeof(int) * num_tiles,
            &attr_size);
        PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoCapGetAttribute);
      }

      // Query the stages enums supported by that algo (cuda must >= 11.0)
      status = dynload::cublasLtMatmulAlgoCapGetAttribute(
          &algo, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &attr_size);
      PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoCapGetAttribute);
      int num_stages = static_cast<int>(attr_size / sizeof(int));
      std::vector<int> stages(num_stages == 0 ? 1 : num_stages);
      if (num_stages == 0) {
        stages[0] = CUBLASLT_MATMUL_STAGES_UNDEFINED;
        num_stages = 1;
      } else {
        status = dynload::cublasLtMatmulAlgoCapGetAttribute(
            &algo,
            CUBLASLT_ALGO_CAP_STAGES_IDS,
            stages.data(),
            sizeof(int) * num_stages,
            &attr_size);
        PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoCapGetAttribute);
      }

      // Retrieve Other Algo Capabilities attributes
      int splitk_support, red_mask, swizzling_max, custom_option_max;
      status = dynload::cublasLtMatmulAlgoCapGetAttribute(
          &algo,
          CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,
          &splitk_support,
          sizeof(splitk_support),
          &attr_size);
      status = dynload::cublasLtMatmulAlgoCapGetAttribute(
          &algo,
          CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK,
          &red_mask,
          sizeof(red_mask),
          &attr_size);
      status = dynload::cublasLtMatmulAlgoCapGetAttribute(
          &algo,
          CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,
          &swizzling_max,
          sizeof(swizzling_max),
          &attr_size);
      status = dynload::cublasLtMatmulAlgoCapGetAttribute(
          &algo,
          CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX,
          &custom_option_max,
          sizeof(custom_option_max),
          &attr_size);
      PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoCapGetAttribute);

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
                status = dynload::cublasLtMatmulAlgoConfigSetAttribute(
                    &algo,
                    CUBLASLT_ALGO_CONFIG_TILE_ID,
                    &tiles[tile_id],
                    sizeof(tiles[tile_id]));
                status = dynload::cublasLtMatmulAlgoConfigSetAttribute(
                    &algo,
                    CUBLASLT_ALGO_CONFIG_STAGES_ID,
                    &stages[stage_id],
                    sizeof(stages[stage_id]));
                status = dynload::cublasLtMatmulAlgoConfigSetAttribute(
                    &algo,
                    CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                    &custom_option,
                    sizeof(custom_option));
                status = dynload::cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k));
                int split_k_val = 1;
                int reduction_scheme = CUBLASLT_REDUCTION_SCHEME_NONE;
                status = dynload::cublasLtMatmulAlgoConfigSetAttribute(
                    &algo,
                    CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                    &split_k_val,
                    sizeof(split_k_val));
                status = dynload::cublasLtMatmulAlgoConfigSetAttribute(
                    &algo,
                    CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                    &reduction_scheme,
                    sizeof(int));
                if (l > 0) {  // Split-K case
                  split_k_val = split_k_candidates[l - 1];
                  status = dynload::cublasLtMatmulAlgoConfigSetAttribute(
                      &algo,
                      CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                      &split_k_candidates[l - 1],
                      sizeof(split_k_candidates[l - 1]));
                  for (reduction_scheme = 1;
                       reduction_scheme <
                           static_cast<int>(CUBLASLT_REDUCTION_SCHEME_MASK) &&
                       (step < limit);
                       reduction_scheme = reduction_scheme << 1) {
                    if (reduction_scheme & red_mask) {
                      status = dynload::cublasLtMatmulAlgoConfigSetAttribute(
                          &algo,
                          CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                          &reduction_scheme,
                          sizeof(reduction_scheme));
                      PADDLE_CUBLASLT_STATUS_CHECK(
                          cublasLtMatmulAlgoConfigSetAttribute);

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
      TestMatmulRun(handle,
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

    int res_id = 0;
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

  ~CublasLtAlgoCache() {
    // Serialize map_ to cache file
    if (search_times_ > 0) {
      int dev;
      cudaGetDevice(&dev);
      if (dev == 0) {
        std::ofstream outfile;
        outfile.open(config_filename_, std::ios::out | std::ios::trunc);
        outfile << dynload::cublasLtGetCudartVersion() << std::endl;

        for (const auto& p : map_) {
          outfile << p.first << " ";
          for (int i = 0; i < 8; ++i) {
            outfile << p.second.data[i] << " ";
          }
          outfile << std::endl;
        }
        outfile.close();
      }
    }
  }

 private:
  explicit CublasLtAlgoCache(int search_times)
      : search_times_(search_times), has_config_file_(true) {
    // Init map_ from cache file
    // do not use cache file
    std::ifstream infile;
    infile.open(config_filename_);
    if (!infile.is_open()) {
      has_config_file_ = false;
      VLOG(3) << "No CublasLtAlgoCache file found";
      return;
    }
    size_t cublaslt_version;
    int64_t seed = 0;
    uint64_t algo_data[8];
    infile >> cublaslt_version;
    VLOG(1) << "cublaslt_version " << cublaslt_version;

    if (dynload::cublasLtGetCudartVersion() != cublaslt_version) {
      LOG(INFO) << config_filename_
                << " is not compatible with current cublaslt_version ";
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

  std::string config_filename_{"/tmp/paddle_cublaslt_cache"};
  std::unordered_map<int64_t, cublasLtMatmulAlgo_t> map_;
  int search_times_;
  const int requested_algo_count_ = 100;
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

  void HashMatmulDesc_(cublasLtMatmulDesc_t desc,
                       int64_t* seed,
                       const std::hash<int64_t>& hash_fn) {
    size_t size_to_write;
    int trans_a, trans_b;
    uint32_t epilogue;

    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulDescGetAttribute(desc,
                                                CUBLASLT_MATMUL_DESC_TRANSA,
                                                &trans_a,
                                                sizeof(trans_a),
                                                &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(trans_a));

    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulDescGetAttribute(desc,
                                                CUBLASLT_MATMUL_DESC_TRANSB,
                                                &trans_b,
                                                sizeof(trans_b),
                                                &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(trans_b));

    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulDescGetAttribute(desc,
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

    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatrixLayoutGetAttribute(desc,
                                                  CUBLASLT_MATRIX_LAYOUT_TYPE,
                                                  &dtype,
                                                  sizeof(dtype),
                                                  &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(dtype));

    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutGetAttribute(
        desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch,
        sizeof(batch),
        &size_to_write));
    HashValue_(seed, hash_fn, static_cast<int64_t>(batch));

    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutGetAttribute(
        desc, CUBLASLT_MATRIX_LAYOUT_ROWS, &row, sizeof(row), &size_to_write));
    HashValue_(seed, hash_fn, RoundToNextHighPowOfTwo(row, 32));

    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutGetAttribute(
        desc, CUBLASLT_MATRIX_LAYOUT_COLS, &col, sizeof(col), &size_to_write));
    HashValue_(seed, hash_fn, RoundToNextHighPowOfTwo(col, 32));

    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutGetAttribute(
        desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld), &size_to_write));
    HashValue_(seed, hash_fn, RoundToNextHighPowOfTwo(ld, 32));

    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatrixLayoutGetAttribute(
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
#endif
}  // namespace cutlass_internal

// Set this enum according to
// https://docs.nvidia.com/cuda/cublas/index.html#cublasltepilogue-t
// While kMatmul, kMatmulGrad, kMatmulGradWithoutBias share the same
// enum value, but if all elements for MatmulPlanner->GetKey() is same,
// no matter forward or backward, they could share the same descriptor
// cache, in that the descriptor is for description of matmul operation.
enum MatmulFusedType {
  kMatmul = 0,
  kMatmulGrad = 1,
  kMatmulGradWithoutBias = 2,
  kMatmulBias = 3,
  kMatmulRelu = 4,
  kMatmulBiasRelu = 5,
  kMatmulBiasGelu = 6,
  kMatmulBiasReluWithReservedData = 7,
  kMatmulBiasGeluWithReservedData = 8,
  kMatmulReluGrad = 9,
  kMatmulGeluGrad = 10,
  kMatmulBiasGradToA = 11,
  kMatmulBiasGradToB = 12
};

static cublasLtEpilogue_t ConvertFusedType(MatmulFusedType fused_type) {
  static std::map<MatmulFusedType, cublasLtEpilogue_t> fused_type_map = {
      {MatmulFusedType::kMatmul, CUBLASLT_EPILOGUE_DEFAULT},
      {MatmulFusedType::kMatmulGrad, CUBLASLT_EPILOGUE_DEFAULT},
      {MatmulFusedType::kMatmulGradWithoutBias, CUBLASLT_EPILOGUE_DEFAULT},
      {MatmulFusedType::kMatmulBias, CUBLASLT_EPILOGUE_BIAS},
      {MatmulFusedType::kMatmulRelu, CUBLASLT_EPILOGUE_RELU},
      {MatmulFusedType::kMatmulBiasRelu, CUBLASLT_EPILOGUE_RELU_BIAS},
      {MatmulFusedType::kMatmulBiasGelu, CUBLASLT_EPILOGUE_GELU_BIAS},
      {MatmulFusedType::kMatmulBiasReluWithReservedData,
       CUBLASLT_EPILOGUE_RELU_AUX_BIAS},
      {MatmulFusedType::kMatmulBiasGeluWithReservedData,
       CUBLASLT_EPILOGUE_GELU_AUX_BIAS},
      {MatmulFusedType::kMatmulReluGrad, CUBLASLT_EPILOGUE_DRELU},
      {MatmulFusedType::kMatmulGeluGrad, CUBLASLT_EPILOGUE_DGELU},
      {MatmulFusedType::kMatmulBiasGradToA, CUBLASLT_EPILOGUE_BGRADA},
      {MatmulFusedType::kMatmulBiasGradToB, CUBLASLT_EPILOGUE_BGRADB}};

  return fused_type_map[fused_type];
}

enum FusedGEMMGradInType { kDX = 0, kDY = 1, kDZ = 2 };

template <bool TransX, bool TransY>
struct FusedGEMMGradTrait;

template <>
struct FusedGEMMGradTrait<false, false> {
  static constexpr auto kXGradA = FusedGEMMGradInType::kDZ;
  static constexpr auto kXGradB = FusedGEMMGradInType::kDY;
  static constexpr auto kXGradATrans = false;
  static constexpr auto kXGradBTrans = true;

  static constexpr auto kYGradA = FusedGEMMGradInType::kDX;
  static constexpr auto kYGradB = FusedGEMMGradInType::kDZ;
  static constexpr auto kYGradATrans = true;
  static constexpr auto kYGradBTrans = false;
};

template <>
struct FusedGEMMGradTrait<true, false> {
  static constexpr auto kXGradA = FusedGEMMGradInType::kDY;
  static constexpr auto kXGradB = FusedGEMMGradInType::kDZ;
  static constexpr auto kXGradATrans = false;
  static constexpr auto kXGradBTrans = true;

  static constexpr auto kYGradA = FusedGEMMGradInType::kDX;
  static constexpr auto kYGradB = FusedGEMMGradInType::kDZ;
  static constexpr auto kYGradATrans = false;
  static constexpr auto kYGradBTrans = false;
};

template <>
struct FusedGEMMGradTrait<false, true> {
  static constexpr auto kXGradA = FusedGEMMGradInType::kDZ;
  static constexpr auto kXGradB = FusedGEMMGradInType::kDY;
  static constexpr auto kXGradATrans = false;
  static constexpr auto kXGradBTrans = false;

  static constexpr auto kYGradA = FusedGEMMGradInType::kDZ;
  static constexpr auto kYGradB = FusedGEMMGradInType::kDX;
  static constexpr auto kYGradATrans = true;
  static constexpr auto kYGradBTrans = false;
};

template <>
struct FusedGEMMGradTrait<true, true> {
  static constexpr auto kXGradA = FusedGEMMGradInType::kDY;
  static constexpr auto kXGradB = FusedGEMMGradInType::kDZ;
  static constexpr auto kXGradATrans = true;
  static constexpr auto kXGradBTrans = true;

  static constexpr auto kYGradA = FusedGEMMGradInType::kDZ;
  static constexpr auto kYGradB = FusedGEMMGradInType::kDX;
  static constexpr auto kYGradATrans = true;
  static constexpr auto kYGradBTrans = true;
};

// To tell any matmul or fused matmul operation from each other.
struct MatmulPlanner {
 public:
  const void* bias{nullptr};
  void* aux_data{nullptr};

  MatmulPlanner() {}
  MatmulPlanner(const std::vector<int64_t>& x_dims,
                const std::vector<int64_t>& y_dims,
                const bool trans_x,
                const bool trans_y,
                phi::DataType dtype,
                MatmulFusedType fused_type,
                const void* bias_data = nullptr,
                void* reserve_data = nullptr,  // Commonly for ReLu bit-mask.
                bool use_addto = false,
                bool no_exchange = true)
      : bias(bias_data), aux_data(reserve_data), fused_type_(fused_type) {
    use_addto_ = use_addto;
    key_ = phi::autotune::GenKey(x_dims,
                                 y_dims,
                                 static_cast<int>(trans_x),
                                 static_cast<int>(trans_y),
                                 static_cast<int>(dtype),
                                 static_cast<int>(fused_type_),
                                 static_cast<int>(use_addto_),
                                 static_cast<int>(no_exchange));
  }

  bool UseAddTo() const { return use_addto_; }
  size_t GetKey() const { return key_; }
  MatmulFusedType GetFusedType() const { return fused_type_; }

  size_t GenSubKey() const { return key_; }

 private:
  MatmulFusedType fused_type_;
  bool use_addto_;
  size_t key_;
};

template <typename T>
cublasComputeType_t GetCudaComputeType() {
  if (std::is_same<T, double>::value) {
    return CUBLAS_COMPUTE_64F;
  } else if (std::is_same<T, int8_t>::value) {
    return CUBLAS_COMPUTE_32I;
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
  cublasLtMatmulAlgo_t* algo{nullptr};
  bool is_cached{false};
  int64_t M_{-1};
  int64_t N_{-1};
  int64_t K_{-1};
  cublasComputeType_t compute_type_;
  cudaDataType_t scale_type_;
  cudaDataType_t x_type_;
  cudaDataType_t y_type_;
  cudaDataType_t out_type_;

  MatmulDescriptor() {}
  MatmulDescriptor(const MatmulDescriptor& obj) {
    algo = obj.algo;
    x_desc = obj.x_desc;
    y_desc = obj.y_desc;
    op_desc = obj.op_desc;
    out_desc = obj.out_desc;
    is_cached = obj.is_cached;
  }

  MatmulDescriptor& operator=(const MatmulDescriptor& obj) {
    algo = obj.algo;
    x_desc = obj.x_desc;
    y_desc = obj.y_desc;
    op_desc = obj.op_desc;
    out_desc = obj.out_desc;
    is_cached = obj.is_cached;

    return *this;
  }

  ~MatmulDescriptor() PADDLE_MAY_THROW {
    if (!is_cached) {
      PADDLE_WARN_GPU_SUCCESS(dynload::cublasLtMatmulDescDestroy(op_desc));
      PADDLE_WARN_GPU_SUCCESS(dynload::cublasLtMatrixLayoutDestroy(y_desc));
      PADDLE_WARN_GPU_SUCCESS(dynload::cublasLtMatrixLayoutDestroy(x_desc));
      PADDLE_WARN_GPU_SUCCESS(dynload::cublasLtMatrixLayoutDestroy(out_desc));
      delete algo;

      op_desc = nullptr;
      x_desc = nullptr;
      y_desc = nullptr;
      out_desc = nullptr;
      algo = nullptr;
    }
  }

  // x_desc, y_desc, op_desc are allocated in heap memory.
  template <typename T, typename DXT, typename DYT, bool TransX, bool TransY>
  void Create(const int64_t M,
              const int64_t N,
              const int64_t K,
              const bool trans_x,
              const bool trans_y,
              phi::funcs::MatmulPlanner* planner,
              const int batch_size = 1,
              const int64_t stride_x = 0,
              const int64_t stride_y = 0,
              const int64_t stride_out = 0,
              bool grad_for_dx = true) {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    cudaDataType_t mat_type = phi::backends::gpu::ToCudaDataType<T>();
    cudaDataType_t out_mat_type = phi::backends::gpu::ToCudaDataType<T>();
    cudaDataType_t scale_type = phi::backends::gpu::ToCudaDataType<MT>();
    cublasComputeType_t compute_type = GetCudaComputeType<T>();

    if (std::is_same<T, int8_t>::value) {
      out_mat_type = phi::backends::gpu::ToCudaDataType<int32_t>();
      scale_type = phi::backends::gpu::ToCudaDataType<int32_t>();
    }

    // Create operation descriptor; see cublasLtMatmulDescAttributes_t for
    // details about defaults; just need to set the transforms for A and B
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type));
    SetFusedEpilogueOpDescriptor(planner, trans_x, trans_y, N);

    // Create matrix descriptors
    CreateMatrixLayout(&x_desc, mat_type, M, K, trans_x);
    CreateMatrixLayout(&y_desc, mat_type, K, N, trans_y);
    CreateMatrixLayout(&out_desc, out_mat_type, M, N, false);

    // Config batch size and stride.
    if (batch_size > 1) {
      SetBatchAndStride(x_desc, batch_size, stride_x);
      SetBatchAndStride(y_desc, batch_size, stride_y);
      SetBatchAndStride(out_desc, batch_size, stride_out);
    }

    M_ = M;
    N_ = N;
    K_ = K;
    compute_type_ = compute_type;
    scale_type_ = scale_type;
    x_type_ = mat_type;
    y_type_ = mat_type;
    out_type_ = out_mat_type;
  }

  cublasLtMatmulAlgo_t* SetAlgo() {
    // while entering this function, the desc shall be cached.
    is_cached = true;
    algo = new cublasLtMatmulAlgo_t;
    return algo;
  }

  void ForceSetAlgo(cublasLtMatmulAlgo_t* new_algo) {
    // while entering this function, the desc shall be cached.
    is_cached = true;
    algo = new_algo;
  }

  template <typename T>
  void SetFusedEpiloguePtr(phi::funcs::MatmulPlanner* planner) {
    if (planner->bias != nullptr) {
      const T* bias_data = static_cast<const T*>(planner->bias);
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_BIAS_POINTER,
          &bias_data,
          sizeof(bias_data)));
    }
    if (planner->aux_data != nullptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
          &(planner->aux_data),
          sizeof(planner->aux_data)));
    }
  }

  std::string GetDescResultString(std::string prefix,
                                  bool has_algo = true) const {
    std::ostringstream out;
    out << prefix << " \n";
#define GET_DESC_DATA_STRING(src)                    \
  do {                                               \
    out << "  " << #src << " = [";                   \
    int num = sizeof((*src)) / sizeof(src->data[0]); \
    for (int i = 0; i < num; ++i) {                  \
      if (i == 0) {                                  \
        out << src->data[i];                         \
      } else {                                       \
        out << ", " << src->data[i];                 \
      }                                              \
    }                                                \
    out << "]\n";                                    \
  } while (0);

    if (has_algo) {
      GET_DESC_DATA_STRING(algo);
    }
    GET_DESC_DATA_STRING(x_desc);
    GET_DESC_DATA_STRING(y_desc);
    GET_DESC_DATA_STRING(out_desc);
    GET_DESC_DATA_STRING(op_desc);
#undef GET_DESC_DATA_STRING
    return out.str();
  }

  void ExchangeXYDesc(bool no_exchange) {}

 protected:
  void SetFusedEpilogueOpDescriptor(phi::funcs::MatmulPlanner* planner,
                                    const bool trans_x,
                                    const bool trans_y,
                                    int64_t lead_dim) {
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
    MatmulFusedType fused_type = planner->GetFusedType();
    if (fused_type != MatmulFusedType::kMatmul) {
      cublasLtEpilogue_t cublaslt_fused_type = ConvertFusedType(fused_type);
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cublasLtMatmulDescSetAttribute(op_desc,
                                                  CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                  &cublaslt_fused_type,
                                                  sizeof(fused_type)));
    }
    if (planner->aux_data) {
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
          &lead_dim,
          sizeof(lead_dim)));
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
};

struct MatmulGradDescriptor : MatmulDescriptor {
 public:
  MatmulGradDescriptor() {}

  template <typename T, typename DXT, typename DYT, bool TransX, bool TransY>
  void Create(const int64_t M,
              const int64_t N,
              const int64_t K,
              const bool trans_x,
              const bool trans_y,
              phi::funcs::MatmulPlanner* planner,
              const int batch_size = 1,
              int64_t stride_x = 0,
              int64_t stride_y = 0,
              int64_t stride_out = 0,
              bool grad_for_dx = true) {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    cudaDataType_t mat_type = phi::backends::gpu::ToCudaDataType<T>();
    cudaDataType_t scale_type = phi::backends::gpu::ToCudaDataType<MT>();
    cublasComputeType_t compute_type = GetCudaComputeType<T>();

    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type));
    this->SetFusedEpilogueOpDescriptor(
        planner, trans_x, trans_y, TransX ? M : K);

    // Create operation descriptor; see cublasLtMatmulDescAttributes_t for
    // details about defaults; just need to set the transforms for A and B
    this->CreateMatrixLayout(&x_desc, mat_type, N, M, true);
    if (grad_for_dx) {
      this->CreateMatrixLayout(&y_desc, mat_type, K, N, TransY);
      this->CreateMatrixLayout(
          &out_desc, phi::backends::gpu::ToCudaDataType<DXT>(), M, K, TransX);
    } else {
      this->CreateMatrixLayout(&y_desc, mat_type, M, K, TransX);
      this->CreateMatrixLayout(
          &out_desc, phi::backends::gpu::ToCudaDataType<DYT>(), K, N, TransY);
    }
  }

  void ExchangeXYDesc(bool no_exchange) {
    if (no_exchange) {
      return;
    }
    auto* temp = y_desc;
    y_desc = x_desc;
    x_desc = temp;
  }
};

template <typename T, typename OutT = T, class MatmulDescT = MatmulDescriptor>
struct CublasLtBase {
 public:
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  static phi::Allocator::AllocationPtr GetWorkspace(const phi::GPUContext& ctx,
                                                    size_t workspace_size) {
    return phi::memory_utils::Alloc(
        ctx.GetPlace(),
        workspace_size,
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  }

  static void RunImpl(const phi::GPUContext& ctx,
                      MatmulDescT* desc,
                      const size_t sub_key,
                      const T* x_ptr,
                      const T* y_ptr,
                      OutT* out_ptr,
                      phi::funcs::MatmulPlanner* planner) {
    MT alpha = static_cast<MT>(1);
    MT beta = planner->UseAddTo() ? static_cast<MT>(1) : static_cast<MT>(0);
    cublasLtHandle_t cublaslt_handle = ctx.cublaslt_handle();

    // NOTE(limingshu): As workspace_size varies from different DL framework,
    // I wonder is there any smarter idea for workspace setting, currently I
    // just followed the settings from the NVIDIA colleague`s setting.
    size_t workspace_size = static_cast<size_t>(4) * 1024 * 1024;
    phi::Allocator::AllocationPtr workspace = GetWorkspace(ctx, workspace_size);

    if (planner != nullptr) {
      if (phi::autotune::AutoTuneStatus::Instance().UseAutoTune() &&
          (!desc->is_cached)) {
        SearchBestAlgo(ctx,
                       cublaslt_handle,
                       desc,
                       static_cast<void*>(&alpha),
                       static_cast<void*>(&beta),
                       y_ptr,
                       x_ptr,
                       out_ptr,
                       workspace->ptr(),
                       workspace_size);
        MatmulDescT* best_desc = new MatmulDescT(*desc);
        VLOG(6) << best_desc->GetDescResultString(
            "[Searched CublasltDescriptor] ");

        auto& cache = phi::autotune::AutoTuneCache::Instance().GetMatmul();
        cache.SetSubKey(sub_key, reinterpret_cast<void*>(best_desc));
      }
    }

    VLOG(7) << desc->GetDescResultString("[Impl CublasltDescriptor] ");
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmul(cublaslt_handle,
                                desc->op_desc,
                                static_cast<void*>(&alpha),
                                y_ptr,
                                desc->y_desc,
                                x_ptr,
                                desc->x_desc,
                                static_cast<void*>(&beta),
                                out_ptr,
                                desc->out_desc,
                                out_ptr,
                                desc->out_desc,
                                desc->algo,
                                workspace->ptr(),
                                workspace_size,
                                ctx.stream()));
  }

  static void SearchBestAlgo(const phi::GPUContext& ctx,
                             const cublasLtHandle_t& lt_handle,
                             MatmulDescT* desc,
                             const void* alpha,
                             const void* beta,
                             const void* y_data,
                             const void* x_data,
                             void* out_data,
                             void* workspace_ptr,
                             size_t workspace_size) {
    cublasLtMatmulPreference_t preference;
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulPreferenceCreate(&preference));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(workspace_size)));

    int returned_results = 0;
    constexpr int requested_algo_count = 10;
    std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results(
        requested_algo_count);
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulAlgoGetHeuristic(lt_handle,
                                                desc->op_desc,
                                                desc->y_desc,
                                                desc->x_desc,
                                                desc->out_desc,
                                                desc->out_desc,
                                                preference,
                                                requested_algo_count,
                                                heuristic_results.data(),
                                                &returned_results));
    PADDLE_ENFORCE_GT(returned_results,
                      0,
                      phi::errors::Unavailable("No GEMM algorithm available."));
    int best_algo_idx = -1;
    if (returned_results == 1 || FLAGS_cublaslt_exhaustive_search_times <= 0) {
      best_algo_idx = 0;
    } else {
      float min_time_cost = std::numeric_limits<float>::max();
      for (int algo_idx = 0; algo_idx < returned_results; ++algo_idx) {
        float cur_time_cost =
            RunAndMeasureAlgo(ctx,
                              lt_handle,
                              desc,
                              alpha,
                              beta,
                              y_data,
                              x_data,
                              out_data,
                              workspace_ptr,
                              workspace_size,
                              &(heuristic_results[algo_idx].algo));
        VLOG(6) << "[MatmulWithCublaslt] algo[" << algo_idx
                << "] time: " << cur_time_cost << " s";

        if ((best_algo_idx == 0 && (1.05 * cur_time_cost < min_time_cost)) ||
            (cur_time_cost < min_time_cost)) {
          best_algo_idx = algo_idx;
          min_time_cost = cur_time_cost;
        }
      }
    }
    VLOG(6) << "[MatmulWithCublaslt] best_algo_idx: " << best_algo_idx;

    cublasLtMatmulAlgo_t* best_algo = desc->SetAlgo();
    *best_algo = heuristic_results[best_algo_idx].algo;
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulPreferenceDestroy(preference));
  }

  static float RunAndMeasureAlgo(const phi::GPUContext& ctx,
                                 const cublasLtHandle_t& lt_handle,
                                 MatmulDescT* desc,
                                 const void* alpha,
                                 const void* beta,
                                 const void* y_data,
                                 const void* x_data,
                                 void* out_data,
                                 void* workspace_ptr,
                                 size_t workspace_size,
                                 cublasLtMatmulAlgo_t* algo) {
    int repeats = FLAGS_cublaslt_exhaustive_search_times;
    if (repeats <= 0) {
      return std::numeric_limits<float>::max();
    }

    phi::GpuTimer timer;
    float time_cost = 0.f;
    const auto& stream = ctx.stream();

    for (int i = 0; i < repeats; ++i) {
      timer.Start(stream);
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmul(lt_handle,
                                                         desc->op_desc,
                                                         alpha,
                                                         y_data,
                                                         desc->y_desc,
                                                         x_data,
                                                         desc->x_desc,
                                                         beta,
                                                         out_data,
                                                         desc->out_desc,
                                                         out_data,
                                                         desc->out_desc,
                                                         algo,
                                                         workspace_ptr,
                                                         workspace_size,
                                                         stream));
      timer.Stop(stream);
      ctx.Wait();
      auto time = timer.ElapsedTime();
      if (i > 0) {
        // Exclude the warmup runtime.
        time_cost += time;
      }
    }
    return (time_cost / (repeats - 1));
  }
};

template <>
struct CublasLtBase<int8_t, int32_t, MatmulDescriptor> {
 public:
  static phi::Allocator::AllocationPtr GetWorkspace(const phi::GPUContext& ctx,
                                                    size_t workspace_size) {
    return phi::memory_utils::Alloc(
        ctx.GetPlace(),
        workspace_size,
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  }

  static void RunImpl(const phi::GPUContext& ctx,
                      MatmulDescriptor* desc,
                      const size_t sub_key,
                      const int8_t* x_ptr,
                      const int8_t* y_ptr,
                      int32_t* out_ptr,
                      phi::funcs::MatmulPlanner* planner) {
    int32_t alpha = 1;
    int32_t beta =
        planner->UseAddTo() ? static_cast<int32_t>(1) : static_cast<int32_t>(0);
    cublasLtHandle_t cublaslt_handle = ctx.cublaslt_handle();

    size_t workspace_size = static_cast<size_t>(4) * 1024 * 1024;
    phi::Allocator::AllocationPtr workspace = GetWorkspace(ctx, workspace_size);

    if (planner != nullptr) {
      if (FLAGS_enable_blaslt_global_search) {
        SearchBestAlgoGlobal(ctx,
                             cublaslt_handle,
                             desc,
                             static_cast<void*>(&alpha),
                             static_cast<void*>(&beta),
                             y_ptr,
                             x_ptr,
                             out_ptr,
                             workspace->ptr(),
                             workspace_size);
      } else if (phi::autotune::AutoTuneStatus::Instance().UseAutoTune() &&
                 (!desc->is_cached)) {
        SearchBestAlgo(ctx,
                       cublaslt_handle,
                       desc,
                       static_cast<void*>(&alpha),
                       static_cast<void*>(&beta),
                       y_ptr,
                       x_ptr,
                       out_ptr,
                       workspace->ptr(),
                       workspace_size);
      }
      MatmulDescriptor* best_desc = new MatmulDescriptor(*desc);
      VLOG(6) << best_desc->GetDescResultString(
          "[Searched CublasltDescriptor] ");

      auto& cache = phi::autotune::AutoTuneCache::Instance().GetMatmul();
      cache.SetSubKey(sub_key, reinterpret_cast<void*>(best_desc));
    }

    VLOG(7) << desc->GetDescResultString("[Impl CublasltDescriptor] ");
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmul(cublaslt_handle,
                                desc->op_desc,
                                static_cast<void*>(&alpha),
                                y_ptr,
                                desc->y_desc,
                                x_ptr,
                                desc->x_desc,
                                static_cast<void*>(&beta),
                                out_ptr,
                                desc->out_desc,
                                out_ptr,
                                desc->out_desc,
                                desc->algo,
                                workspace->ptr(),
                                workspace_size,
                                ctx.stream()));
  }

  static void SearchBestAlgoGlobal(const phi::GPUContext& ctx,
                                   const cublasLtHandle_t& lt_handle,
                                   MatmulDescriptor* desc,
                                   const void* alpha,
                                   const void* beta,
                                   const void* y_data,
                                   const void* x_data,
                                   void* out_data,
                                   void* workspace_ptr,
                                   size_t& workspace_size) {  // NOLINT
    void* bias_ptr = nullptr;
    cublasLtMatmulAlgo_t* algo =
        cutlass_internal::CublasLtAlgoCache::Instance().CublasLtAlgoSelect(
            ctx.cublaslt_handle(),
            desc->M_,
            desc->N_,
            desc->K_,
            1,
            y_data,
            x_data,
            bias_ptr,
            out_data,
            const_cast<void*>(alpha),
            const_cast<void*>(beta),
            desc->op_desc,
            desc->y_desc,
            desc->x_desc,
            desc->out_desc,
            desc->out_desc,
            desc->compute_type_,
            desc->scale_type_,
            desc->y_type_,
            desc->x_type_,
            desc->out_type_,
            desc->out_type_,
            ctx.stream());
    if (algo == nullptr) {
      LOG(WARNING) << "CublasLtAlgoSelect failed, result is empty! We attempt "
                      "to use Heuristic search";
      cublasLtMatmulPreference_t preference;
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cublasLtMatmulPreferenceCreate(&preference));
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulPreferenceSetAttribute(
          preference,
          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &workspace_size,
          sizeof(workspace_size)));

      int returned_results = 0;
      constexpr int requested_algo_count = 1;
      std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results(
          requested_algo_count);
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cublasLtMatmulAlgoGetHeuristic(lt_handle,
                                                  desc->op_desc,
                                                  desc->y_desc,
                                                  desc->x_desc,
                                                  desc->out_desc,
                                                  desc->out_desc,
                                                  preference,
                                                  requested_algo_count,
                                                  heuristic_results.data(),
                                                  &returned_results));
      PADDLE_ENFORCE_GT(
          returned_results,
          0,
          phi::errors::Unavailable("No GEMM algorithm available."));
      algo = &heuristic_results[0].algo;
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cublasLtMatmulPreferenceDestroy(preference));
    }
    cublasLtMatmulHeuristicResult_t heurResult;
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulAlgoCheck(ctx.cublaslt_handle(),
                                         desc->op_desc,
                                         desc->y_desc,
                                         desc->x_desc,
                                         desc->out_desc,
                                         desc->out_desc,
                                         algo,
                                         &heurResult));
    desc->ForceSetAlgo(algo);
    size_t temp_workspace_size = heurResult.workspaceSize;
    auto temp_workspace = phi::memory_utils::Alloc(
        phi::GPUPlace(backends::gpu::GetCurrentDeviceId()),
        temp_workspace_size);
    workspace_ptr = temp_workspace->ptr();
    workspace_size = temp_workspace_size;
  }

  static void SearchBestAlgo(const phi::GPUContext& ctx,
                             const cublasLtHandle_t& lt_handle,
                             MatmulDescriptor* desc,
                             const void* alpha,
                             const void* beta,
                             const void* y_data,
                             const void* x_data,
                             void* out_data,
                             void* workspace_ptr,
                             size_t workspace_size) {
    cublasLtMatmulPreference_t preference;
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulPreferenceCreate(&preference));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(workspace_size)));

    int returned_results = 0;
    constexpr int requested_algo_count = 10;
    std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results(
        requested_algo_count);
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulAlgoGetHeuristic(lt_handle,
                                                desc->op_desc,
                                                desc->y_desc,
                                                desc->x_desc,
                                                desc->out_desc,
                                                desc->out_desc,
                                                preference,
                                                requested_algo_count,
                                                heuristic_results.data(),
                                                &returned_results));
    PADDLE_ENFORCE_GT(returned_results,
                      0,
                      phi::errors::Unavailable("No GEMM algorithm available."));
    int best_algo_idx = -1;
    if (returned_results == 1 || FLAGS_cublaslt_exhaustive_search_times <= 0) {
      best_algo_idx = 0;
    } else {
      float min_time_cost = std::numeric_limits<float>::max();
      for (int algo_idx = 0; algo_idx < returned_results; ++algo_idx) {
        float cur_time_cost =
            RunAndMeasureAlgo(ctx,
                              lt_handle,
                              desc,
                              alpha,
                              beta,
                              y_data,
                              x_data,
                              out_data,
                              workspace_ptr,
                              workspace_size,
                              &(heuristic_results[algo_idx].algo));
        VLOG(6) << "[MatmulWithCublaslt] algo[" << algo_idx
                << "] time: " << cur_time_cost << " s";

        if ((best_algo_idx == 0 && (1.05 * cur_time_cost < min_time_cost)) ||
            (cur_time_cost < min_time_cost)) {
          best_algo_idx = algo_idx;
          min_time_cost = cur_time_cost;
        }
      }
    }
    VLOG(6) << "[MatmulWithCublaslt] best_algo_idx: " << best_algo_idx;

    cublasLtMatmulAlgo_t* best_algo = desc->SetAlgo();
    *best_algo = heuristic_results[best_algo_idx].algo;
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::cublasLtMatmulPreferenceDestroy(preference));
  }

  static float RunAndMeasureAlgo(const phi::GPUContext& ctx,
                                 const cublasLtHandle_t& lt_handle,
                                 MatmulDescriptor* desc,
                                 const void* alpha,
                                 const void* beta,
                                 const void* y_data,
                                 const void* x_data,
                                 void* out_data,
                                 void* workspace_ptr,
                                 size_t workspace_size,
                                 cublasLtMatmulAlgo_t* algo) {
    int repeats = FLAGS_cublaslt_exhaustive_search_times;
    if (repeats <= 0) {
      return std::numeric_limits<float>::max();
    }

    phi::GpuTimer timer;
    float time_cost = 0.f;
    const auto& stream = ctx.stream();

    for (int i = 0; i < repeats; ++i) {
      timer.Start(stream);
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cublasLtMatmul(lt_handle,
                                                         desc->op_desc,
                                                         alpha,
                                                         y_data,
                                                         desc->y_desc,
                                                         x_data,
                                                         desc->x_desc,
                                                         beta,
                                                         out_data,
                                                         desc->out_desc,
                                                         out_data,
                                                         desc->out_desc,
                                                         algo,
                                                         workspace_ptr,
                                                         workspace_size,
                                                         stream));
      timer.Stop(stream);
      ctx.Wait();
      auto time = timer.ElapsedTime();
      if (i > 0) {
        // Exclude the warmup runtime.
        time_cost += time;
      }
    }
    return (time_cost / (repeats - 1));
  }
};

// To judge if desc is cached or not.
template <class DescT,
          typename T,
          typename DXT = T,
          typename DYT = T,
          bool TransX = false,
          bool TransY = false>
struct DescriptorSetter {
 public:
  DescT desc;
  size_t sub_key{std::numeric_limits<size_t>::min()};

  DescriptorSetter(phi::funcs::MatmulPlanner* planner,
                   const int64_t M,
                   const int64_t N,
                   const int64_t K,
                   const bool trans_x,
                   const bool trans_y,
                   const int batch_size = 1,
                   int64_t stride_x = 0,
                   int64_t stride_y = 0,
                   int64_t stride_out = 0,
                   const bool no_exchange = true,
                   bool grad_for_dx = true) {
    if (std::is_same<T, int8_t>::value) {
      if (!trans_x && !trans_y) {
        PADDLE_ENFORCE_EQ(
            (N % 4 == 0 || N == 1),
            true,
            phi::errors::InvalidArgument(
                "The dimension size N used in int8 matmul must be 1 or a "
                "multiple of 4 does not "
                "match the size (%d) currently contained in the container.",
                N));
        PADDLE_ENFORCE_EQ(
            (K % 4 == 0),
            true,
            phi::errors::InvalidArgument(
                "The dimension size K used in int8 matmul must be a multiple "
                "of 4 does not "
                "match the size (%d) currently contained in the container.",
                K));
      } else if (!trans_x && trans_y) {
        PADDLE_ENFORCE_EQ(
            (K % 4 == 0),
            true,
            phi::errors::InvalidArgument(
                "The dimension size K used in int8 matmul must be a multiple "
                "of 4 does not "
                "match the size (%d) currently contained in the container.",
                K));
      } else if (trans_x && !trans_y) {
        PADDLE_ENFORCE_EQ(
            (M % 4 == 0 || M == 1),
            true,
            phi::errors::InvalidArgument(
                "The dimension size M used in int8 matmul must be 1 or a "
                "multiple of 4 does not "
                "match the size (%d) currently contained in the container.",
                M));
        PADDLE_ENFORCE_EQ(
            (N % 4 == 0 || N == 1),
            true,
            phi::errors::InvalidArgument(
                "The dimension size N used in int8 matmul must be 1 or a "
                "multiple of 4 does not "
                "match the size (%d) currently contained in the container.",
                N));
      } else {
        PADDLE_ENFORCE_EQ(
            (M % 4 == 0 || M == 1),
            true,
            phi::errors::InvalidArgument(
                "The dimension size M used in int8 matmul must be 1 or a "
                "multiple of 4 does not "
                "match the size (%d) currently contained in the container.",
                M));
        PADDLE_ENFORCE_EQ(
            (K % 4 == 0),
            true,
            phi::errors::InvalidArgument(
                "The dimension size K used in int8 matmul must be a multiple "
                "of 4 does not "
                "match the size (%d) currently contained in the container.",
                K));
      }
    }

    if (planner != nullptr) {
      sub_key = planner->GenSubKey();
    }

    auto& matmul_cache = phi::autotune::AutoTuneCache::Instance().GetMatmul();
    if (matmul_cache.FindSubKey(sub_key)) {
      desc = *(reinterpret_cast<DescT*>(matmul_cache.GetSubKey(sub_key)));
      desc.template SetFusedEpiloguePtr<DYT>(planner);
      VLOG(7) << desc.GetDescResultString("[Heap CublasltDescriptor] ");
    } else {
      desc.template Create<T, DXT, DYT, TransX, TransY>(M,
                                                        N,
                                                        K,
                                                        trans_x,
                                                        trans_y,
                                                        planner,
                                                        batch_size,
                                                        stride_x,
                                                        stride_y,
                                                        stride_out,
                                                        grad_for_dx);
      desc.ExchangeXYDesc(no_exchange);
      if (planner != nullptr) {
        desc.template SetFusedEpiloguePtr<DYT>(planner);
      }
      VLOG(7) << desc.GetDescResultString("[Stack CublasltDescriptor] ", false);
    }
  }
};

// For matmul with kernels autotune
template <typename T, typename OutT = T>
struct MatmulWithCublasLt : public CublasLtBase<T, OutT> {
 public:
  static void Run(const phi::GPUContext& ctx,
                  const T* x_data,
                  const T* y_data,
                  OutT* out_data,
                  const int64_t M,
                  const int64_t N,
                  const int64_t K,
                  const bool trans_x,
                  const bool trans_y,
                  phi::funcs::MatmulPlanner* planner = nullptr) {
    auto setter = DescriptorSetter<MatmulDescriptor, T>(
        planner, M, N, K, trans_x, trans_y);
    CublasLtBase<T, OutT>::RunImpl(
        ctx, &setter.desc, setter.sub_key, x_data, y_data, out_data, planner);
  }

  static void RunWithBatch(const phi::GPUContext& ctx,
                           const T* x_data,
                           const T* y_data,
                           OutT* out_data,
                           const int64_t M,
                           const int64_t N,
                           const int64_t K,
                           bool trans_x,
                           bool trans_y,
                           int batch_size,
                           int64_t stride_x,
                           int64_t stride_y,
                           int64_t stride_out,
                           phi::funcs::MatmulPlanner* planner = nullptr) {
    auto setter = DescriptorSetter<MatmulDescriptor, T>(planner,
                                                        M,
                                                        N,
                                                        K,
                                                        trans_x,
                                                        trans_y,
                                                        batch_size,
                                                        stride_x,
                                                        stride_y,
                                                        stride_out);
    CublasLtBase<T, OutT>::RunImpl(
        ctx, &setter.desc, setter.sub_key, x_data, y_data, out_data, planner);
  }

  static void RunWithBatch(const phi::GPUContext& ctx,
                           const T** x_data,
                           const T** y_data,
                           OutT** out_data,
                           const int64_t M,
                           const int64_t N,
                           const int64_t K,
                           bool trans_x,
                           bool trans_y,
                           int batch_size,
                           phi::funcs::MatmulPlanner* planner = nullptr) {
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
          planner);
    }
  }
};

// As for just Linear fused ephilogue below: out = matmul(x, y) + bias.
template <typename T>
struct LinearWithCublasLt : public CublasLtBase<T> {
  static void Run(const phi::GPUContext& ctx,
                  const phi::DenseTensor* x,
                  const phi::DenseTensor* y,
                  phi::DenseTensor* out,
                  const void* bias_data,
                  void* reserve_data,
                  const int64_t M,
                  const int64_t N,
                  const int64_t K,
                  const bool trans_x,
                  const bool trans_y,
                  const MatmulFusedType fused_type) {
    auto planner = phi::funcs::MatmulPlanner(common::vectorize(x->dims()),
                                             common::vectorize(y->dims()),
                                             trans_x,
                                             trans_y,
                                             phi::CppTypeToDataType<T>::Type(),
                                             fused_type,
                                             bias_data,
                                             reserve_data);
    auto setter = DescriptorSetter<MatmulDescriptor, T>(
        &planner, M, N, K, trans_x, trans_y);
    CublasLtBase<T>::RunImpl(ctx,
                             &setter.desc,
                             setter.sub_key,
                             x->data<T>(),
                             y->data<T>(),
                             out->data<T>(),
                             &planner);
  }
};

template <typename T, typename DXT, typename DYT, bool TransX, bool TransY>
struct LinearGradWithCublasLt : public CublasLtBase<T> {
  static void Run(
      const phi::GPUContext& ctx,
      const phi::DenseTensor* x,
      const phi::DenseTensor* y,
      phi::DenseTensor* out,
      const void* bias_data,
      void* reserve_data,
      const int64_t M,
      const int64_t N,
      const int64_t K,
      const MatmulFusedType fused_type,
      const bool trans_x,
      const bool trans_y,
      const bool use_addto,
      const bool no_exchange,  // exchange x_desc and y_desc for grad.
      bool grad_for_dx = true) {
    auto planner = phi::funcs::MatmulPlanner(common::vectorize(x->dims()),
                                             common::vectorize(y->dims()),
                                             trans_x,
                                             trans_y,
                                             phi::CppTypeToDataType<T>::Type(),
                                             fused_type,
                                             bias_data,
                                             reserve_data,
                                             use_addto,
                                             no_exchange);
    auto setter =
        DescriptorSetter<MatmulGradDescriptor, T, DXT, DYT, TransX, TransY>(
            &planner,
            M,
            N,
            K,
            trans_x,
            trans_y,
            /*batch_size=*/1,
            /*stride_x=*/0,
            /*stride_y=*/0,
            /*stride_out=*/0,
            /*exchange_x_y_desc=*/no_exchange,
            /*grad_for_dx=*/grad_for_dx);

    // To setting data type for different kinda out_data.
    if (grad_for_dx) {
      CublasLtBase<T, DXT, MatmulGradDescriptor>::RunImpl(
          ctx,
          &setter.desc,
          setter.sub_key,
          no_exchange ? x->data<T>() : y->data<T>(),
          no_exchange ? y->data<T>() : x->data<T>(),
          out->data<DXT>(),
          &planner);
    } else {
      CublasLtBase<T, DYT, MatmulGradDescriptor>::RunImpl(
          ctx,
          &setter.desc,
          setter.sub_key,
          no_exchange ? x->data<T>() : y->data<T>(),
          no_exchange ? y->data<T>() : x->data<T>(),
          out->data<DYT>(),
          &planner);
    }
  }
};
#else
// A void structure just for successfully compile.
struct MatmulPlanner {};
#endif  // (PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060

}  // namespace funcs
}  // namespace phi
