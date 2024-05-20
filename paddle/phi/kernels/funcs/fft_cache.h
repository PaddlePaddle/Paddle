// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/kernels/funcs/cufft_util.h"
#elif defined(PADDLE_WITH_HIP)
#include "paddle/phi/kernels/funcs/hipfft_util.h"
#endif

namespace phi {
namespace funcs {
namespace detail {

#if CUDA_VERSION < 10000
// Note that the max plan number for CUDA version < 10 has to be 1023
// due to a bug that fails on the 1024th plan
constexpr size_t CUFFT_MAX_PLAN_NUM = 1023;
constexpr size_t CUFFT_DEFAULT_CACHE_SIZE = CUFFT_MAX_PLAN_NUM;
#else
constexpr size_t CUFFT_MAX_PLAN_NUM = std::numeric_limits<size_t>::max();
// The default max cache size chosen for CUDA version > 10 is arbitrary.
// This number puts a limit on how big of a plan cache should we maintain by
// default. Users can always configure it via cufft_set_plan_cache_max_size.
constexpr size_t CUFFT_DEFAULT_CACHE_SIZE = 4096;
#endif

static_assert(CUFFT_MAX_PLAN_NUM >= 0 &&
                  CUFFT_MAX_PLAN_NUM <= std::numeric_limits<size_t>::max(),
              "CUFFT_MAX_PLAN_NUM not in size_t range");
static_assert(CUFFT_DEFAULT_CACHE_SIZE >= 0 &&
                  CUFFT_DEFAULT_CACHE_SIZE <= CUFFT_MAX_PLAN_NUM,
              "CUFFT_DEFAULT_CACHE_SIZE not in [0, CUFFT_MAX_PLAN_NUM] range");

class FFTConfigCache {
 public:
  using kv_t = typename std::pair<FFTConfigKey, FFTConfig>;
  using map_t =
      typename std::unordered_map<std::reference_wrapper<FFTConfigKey>,
                                  typename std::list<kv_t>::iterator,
                                  KeyHash<FFTConfigKey>,
                                  KeyEqual<FFTConfigKey>>;
  using map_kkv_iter_t = typename map_t::iterator;

  FFTConfigCache() : FFTConfigCache(CUFFT_DEFAULT_CACHE_SIZE) {}

  explicit FFTConfigCache(int64_t max_size) { _set_max_size(max_size); }

  FFTConfigCache(const FFTConfigCache& other) = delete;
  FFTConfigCache& operator=(const FFTConfigCache& other) = delete;

  FFTConfigCache(FFTConfigCache&& other) noexcept
      : _usage_list(std::move(other._usage_list)),
        _cache_map(std::move(other._cache_map)),
        _max_size(other._max_size) {}

  FFTConfigCache& operator=(FFTConfigCache&& other) noexcept {
    _usage_list = std::move(other._usage_list);
    _cache_map = std::move(other._cache_map);
    _max_size = other._max_size;
    return *this;
  }

  // If key is in this cache, return the cached config. Otherwise, emplace the
  // config in this cache and return it.
  FFTConfig& lookup(FFTConfigKey params) {
    PADDLE_ENFORCE_GT(_max_size,
                      0,
                      phi::errors::InvalidArgument(
                          "The max size of FFTConfigCache must be great than 0,"
                          "But received is [%d]",
                          _max_size));

    map_kkv_iter_t map_it = _cache_map.find(params);
    // Hit, put to list front
    if (map_it != _cache_map.end()) {
      _usage_list.splice(_usage_list.begin(), _usage_list, map_it->second);
      return map_it->second->second;
    }

    // Miss
    // remove if needed
    if (_usage_list.size() >= _max_size) {
      auto last = _usage_list.end();
      last--;
      _cache_map.erase(last->first);
      _usage_list.pop_back();
    }

    // construct new plan at list front, then insert into _cache_map
    _usage_list.emplace_front(std::piecewise_construct,
                              std::forward_as_tuple(params),
                              std::forward_as_tuple(params));
    auto kv_it = _usage_list.begin();
    _cache_map.emplace(std::piecewise_construct,
                       std::forward_as_tuple(kv_it->first),
                       std::forward_as_tuple(kv_it));
    return kv_it->second;
  }

  void clear() {
    _cache_map.clear();
    _usage_list.clear();
  }

  void resize(int64_t new_size) {
    _set_max_size(new_size);
    auto cur_size = _usage_list.size();
    if (cur_size > _max_size) {
      auto delete_it = _usage_list.end();
      for (size_t i = 0; i < cur_size - _max_size; i++) {
        delete_it--;
        _cache_map.erase(delete_it->first);
      }
      _usage_list.erase(delete_it, _usage_list.end());
    }
  }

  size_t size() const { return _cache_map.size(); }

  size_t max_size() const noexcept { return _max_size; }

  std::mutex mutex;

 private:
  // Only sets size and does value check. Does not resize the data structures.
  void _set_max_size(int64_t new_size) {
    // We check that 0 <= new_size <= CUFFT_MAX_PLAN_NUM here. Since
    // CUFFT_MAX_PLAN_NUM is of type size_t, we need to do non-negativity check
    // first.
    PADDLE_ENFORCE_GE(
        new_size,
        0,
        phi::errors::InvalidArgument(
            "cuFFT plan cache size must be non-negative, But received is [%d]",
            new_size));
    PADDLE_ENFORCE_LE(new_size,
                      CUFFT_MAX_PLAN_NUM,
                      phi::errors::InvalidArgument(
                          "cuFFT plan cache size can not be larger than [%d], "
                          "But received is [%d]",
                          CUFFT_MAX_PLAN_NUM,
                          new_size));
    _max_size = static_cast<size_t>(new_size);
  }

  std::list<kv_t> _usage_list;
  map_t _cache_map;
  size_t _max_size;
};

static std::vector<std::unique_ptr<FFTConfigCache>> plan_caches;
static std::mutex plan_caches_mutex;

static inline FFTConfigCache& get_fft_plan_cache(int64_t device_index) {
  std::lock_guard<std::mutex> guard(plan_caches_mutex);

  if (device_index >= plan_caches.size()) {
    plan_caches.resize(device_index + 1);
  }

  if (!plan_caches[device_index]) {
    plan_caches[device_index] = std::make_unique<FFTConfigCache>();
  }

  return *plan_caches[device_index];
}
}  // namespace detail
}  // namespace funcs
}  // namespace phi
