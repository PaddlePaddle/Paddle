/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <functional>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/platform/cudnn_helper.h"

DECLARE_uint64(conv_workspace_size_limit);
DECLARE_bool(cudnn_exhaustive_search);
DECLARE_int64(cudnn_exhaustive_search_times);

namespace paddle {
namespace operators {

static constexpr char kCUDNNFwdAlgoCache[] = "kCUDNNFwdAlgoCache";
static constexpr char kCUDNNBwdDataAlgoCache[] = "kCUDNNBwdDataAlgoCache";
static constexpr char kCUDNNBwdFilterAlgoCache[] = "kCUDNNBwdFilterAlgoCache";

static constexpr size_t kCONV_CUDNN_WORKSPACE_LIMIT_BYTES =
    static_cast<size_t>(1024) * 1024 * 1024;

#if CUDNN_VERSION_MIN(6, 0, 5)
static constexpr size_t kNUM_CUDNN_FWD_ALGS = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
static constexpr size_t kNUM_CUDNN_BWD_FILTER_ALGS =
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
static constexpr size_t kNUM_CUDNN_BWD_DATA_ALGS =
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
#else
// cuDNN v5 has no CUDNN_CONVOLUTION_FWD_ALGO_COUNT etc.
static constexpr size_t kNUM_CUDNN_FWD_ALGS = 7;
static constexpr size_t kNUM_CUDNN_BWD_FILTER_ALGS = 4;
static constexpr size_t kNUM_CUDNN_BWD_DATA_ALGS = 5;
#endif

template <typename TAlgorithm>
class AlgorithmsCache {
 public:
  AlgorithmsCache() : search_times_(0) { hash_.clear(); }
  // Caches the best algorithm for a given
  // combination of tensor dimensions & compute data type.
  TAlgorithm GetAlgorithm(
      const std::vector<int64_t>& dims1, const std::vector<int64_t>& dims2,
      const std::vector<int>& strides, const std::vector<int>& paddings,
      const std::vector<int>& dilations,
      int algorithmFlags,  // can set for different data type
      std::function<TAlgorithm()> gen_func);

  TAlgorithm GetAlgorithm(int64_t area, int search_times, int algorithmFlags,
                          std::function<TAlgorithm()> gen_func);

 private:
  std::unordered_map<int64_t, TAlgorithm> hash_;
  std::mutex mutex_;

  int search_times_;
};

template <typename TAlgorithm>
TAlgorithm AlgorithmsCache<TAlgorithm>::GetAlgorithm(
    const std::vector<int64_t>& dims1, const std::vector<int64_t>& dims2,
    const std::vector<int>& strides, const std::vector<int>& paddings,
    const std::vector<int>& dilations, int algorithmFlags,
    std::function<TAlgorithm()> gen_func) {
  std::lock_guard<std::mutex> lock(mutex_);
  int64_t seed = 0;
  // Hash all of the inputs, use to try and look up a previously
  // discovered algorithm, or fall back to generating a new one.
  std::hash<int64_t> hashFn;
  // do hash like boost
  // https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
  for (const auto num : dims1) {
    seed ^= hashFn(num) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  for (const auto num : dims2) {
    seed ^= hashFn(num) + 0x9e3779b9 + (seed << 6) + (seed >> 2) + 1;
  }

  for (const auto num : strides) {
    seed ^= hashFn(static_cast<int64_t>(num)) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2) + 2;
  }

  for (const auto num : paddings) {
    seed ^= hashFn(static_cast<int64_t>(num)) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2) + 3;
  }

  for (const auto num : dilations) {
    seed ^= hashFn(static_cast<int64_t>(num)) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2) + 4;
  }

  seed ^= hashFn(static_cast<int64_t>(algorithmFlags)) + 0x9e3779b9 +
          (seed << 6) + (seed >> 2) + 5;

  if (seed == 0) return gen_func();

  if (hash_.find(seed) == hash_.end()) {
    TAlgorithm value = gen_func();
    hash_[seed] = value;
  }
  return hash_[seed];
}

template <typename TAlgorithm>
TAlgorithm AlgorithmsCache<TAlgorithm>::GetAlgorithm(
    int64_t area, int search_times, int algorithmFlags,
    std::function<TAlgorithm()> gen_func) {
  if (hash_.find(area) != hash_.end()) {
    return hash_[area];
  }
  if (search_times_ < search_times) {
    auto algo = gen_func();
    hash_[area] = algo;
    ++search_times_;
    return algo;
  }
  TAlgorithm algo;
  int64_t min = static_cast<uint64_t>(INT_MAX);
  for (const auto& m : hash_) {
    if (m.first < min) {
      min = m.first;
      algo = m.second;
    }
  }
  return algo;
}

}  // namespace operators
}  // namespace paddle
