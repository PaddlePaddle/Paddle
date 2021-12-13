// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/spectral_op.h"

#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/dynload/hipfft.h"
#endif

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/cufft.h"
#endif

namespace paddle {
namespace operators {
using ScalarType = framework::proto::VarType::Type;
const int64_t kMaxFFTNdim = 3;
const int64_t kMaxDataNdim = kMaxFFTNdim + 1;
// This struct is used to easily compute hashes of the
// parameters. It will be the **key** to the plan cache.
struct FFTConfigKey {
  // between 1 and kMaxFFTNdim, i.e., 1 <= signal_ndim <= 3
  int64_t signal_ndim_;
  // These include additional batch dimension as well.
  int64_t sizes_[kMaxDataNdim];
  int64_t input_shape_[kMaxDataNdim];
  int64_t output_shape_[kMaxDataNdim];
  FFTTransformType fft_type_;
  ScalarType value_type_;

  FFTConfigKey() = default;

  FFTConfigKey(const std::vector<int64_t>& in_shape,
               const std::vector<int64_t>& out_shape,
               const std::vector<int64_t>& signal_size,
               FFTTransformType fft_type, ScalarType value_type) {
    // Padding bits must be zeroed for hashing
    memset(this, 0, sizeof(*this));
    signal_ndim_ = signal_size.size() - 1;
    fft_type_ = fft_type;
    value_type_ = value_type;

    std::copy(signal_size.cbegin(), signal_size.cend(), sizes_);
    std::copy(in_shape.cbegin(), in_shape.cend(), input_shape_);
    std::copy(out_shape.cbegin(), out_shape.cend(), output_shape_);
  }
};

#if defined(PADDLE_WITH_CUDA)
// An RAII encapsulation of cuFFTHandle
class CuFFTHandle {
  ::cufftHandle handle_;

 public:
  CuFFTHandle() {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cufftCreate(&handle_));
  }

  CuFFTHandle(const CuFFTHandle& other) = delete;
  CuFFTHandle& operator=(const CuFFTHandle& other) = delete;

  CuFFTHandle(CuFFTHandle&& other) = delete;
  CuFFTHandle& operator=(CuFFTHandle&& other) = delete;

  ::cufftHandle& get() { return handle_; }
  const ::cufftHandle& get() const { return handle_; }

  ~CuFFTHandle() {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cufftDestroy(handle_));
  }
};

using plan_size_type = long long int;  // NOLINT
// This class contains all the information needed to execute a cuFFT plan:
//   1. the plan
//   2. the workspace size needed
class FFTConfig {
 public:
  // Only move semantics is enought for this class. Although we already use
  // unique_ptr for the plan, still remove copy constructor and assignment op so
  // we don't accidentally copy and take perf hit.
  explicit FFTConfig(const FFTConfigKey& plan_key)
      : FFTConfig(
            std::vector<int64_t>(plan_key.sizes_,
                                 plan_key.sizes_ + plan_key.signal_ndim_ + 1),
            plan_key.signal_ndim_, plan_key.fft_type_, plan_key.value_type_) {}

  // sizes are full signal, including batch size and always two-sided
  FFTConfig(const std::vector<int64_t>& sizes, const int64_t signal_ndim,
            FFTTransformType fft_type, ScalarType dtype)
      : fft_type_(fft_type), value_type_(dtype) {
    // signal sizes (excluding batch dim)
    std::vector<plan_size_type> signal_sizes(sizes.begin() + 1, sizes.end());

    // input batch size
    const auto batch = static_cast<plan_size_type>(sizes[0]);
    // const int64_t signal_ndim = sizes.size() - 1;
    PADDLE_ENFORCE_EQ(signal_ndim, sizes.size() - 1,
                      platform::errors::InvalidArgument(
                          "The signal_ndim must be equal to sizes.size() - 1,"
                          "But signal_ndim is: [%d], sizes.size() - 1 is: [%d]",
                          signal_ndim, sizes.size() - 1));

    cudaDataType itype, otype, exec_type;
    const auto complex_input = has_complex_input(fft_type);
    const auto complex_output = has_complex_output(fft_type);
    if (dtype == framework::proto::VarType::FP32) {
      itype = complex_input ? CUDA_C_32F : CUDA_R_32F;
      otype = complex_output ? CUDA_C_32F : CUDA_R_32F;
      exec_type = CUDA_C_32F;
    } else if (dtype == framework::proto::VarType::FP64) {
      itype = complex_input ? CUDA_C_64F : CUDA_R_64F;
      otype = complex_output ? CUDA_C_64F : CUDA_R_64F;
      exec_type = CUDA_C_64F;
    } else if (dtype == framework::proto::VarType::FP16) {
      itype = complex_input ? CUDA_C_16F : CUDA_R_16F;
      otype = complex_output ? CUDA_C_16F : CUDA_R_16F;
      exec_type = CUDA_C_16F;
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "cuFFT only support transforms of type float16, float32 and "
          "float64"));
    }

    // disable auto allocation of workspace to use allocator from the framework
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cufftSetAutoAllocation(
        plan(), /* autoAllocate */ 0));

    size_t ws_size_t;

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cufftXtMakePlanMany(
        plan(), signal_ndim, signal_sizes.data(),
        /* inembed */ nullptr, /* base_istride */ 1, /* idist */ 1, itype,
        /* onembed */ nullptr, /* base_ostride */ 1, /* odist */ 1, otype,
        batch, &ws_size_t, exec_type));

    ws_size = ws_size_t;
  }

  FFTConfig(const FFTConfig& other) = delete;
  FFTConfig& operator=(const FFTConfig& other) = delete;

  FFTConfig(FFTConfig&& other) = delete;
  FFTConfig& operator=(FFTConfig&& other) = delete;

  const cufftHandle& plan() const { return plan_ptr.get(); }

  FFTTransformType transform_type() const { return fft_type_; }
  ScalarType data_type() const { return value_type_; }
  size_t workspace_size() const { return ws_size; }

 private:
  CuFFTHandle plan_ptr;
  size_t ws_size;
  FFTTransformType fft_type_;
  ScalarType value_type_;
};

#elif defined(PADDLE_WITH_HIP)
// An RAII encapsulation of cuFFTHandle
class HIPFFTHandle {
  ::hipfftHandle handle_;

 public:
  HIPFFTHandle() {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::hipfftCreate(&handle_));
  }

  HIPFFTHandle(const HIPFFTHandle& other) = delete;
  HIPFFTHandle& operator=(const HIPFFTHandle& other) = delete;

  HIPFFTHandle(HIPFFTHandle&& other) = delete;
  HIPFFTHandle& operator=(HIPFFTHandle&& other) = delete;

  ::hipfftHandle& get() { return handle_; }
  const ::hipfftHandle& get() const { return handle_; }

  ~HIPFFTHandle() {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::hipfftDestroy(handle_));
  }
};
using plan_size_type = int;
// This class contains all the information needed to execute a cuFFT plan:
//   1. the plan
//   2. the workspace size needed
class FFTConfig {
 public:
  // Only move semantics is enought for this class. Although we already use
  // unique_ptr for the plan, still remove copy constructor and assignment op so
  // we don't accidentally copy and take perf hit.
  explicit FFTConfig(const FFTConfigKey& plan_key)
      : FFTConfig(
            std::vector<int64_t>(plan_key.sizes_,
                                 plan_key.sizes_ + plan_key.signal_ndim_ + 1),
            plan_key.signal_ndim_, plan_key.fft_type_, plan_key.value_type_) {}

  // sizes are full signal, including batch size and always two-sided
  FFTConfig(const std::vector<int64_t>& sizes, const int64_t signal_ndim,
            FFTTransformType fft_type, ScalarType dtype)
      : fft_type_(fft_type), value_type_(dtype) {
    // signal sizes (excluding batch dim)
    std::vector<plan_size_type> signal_sizes(sizes.begin() + 1, sizes.end());

    // input batch size
    const auto batch = static_cast<plan_size_type>(sizes[0]);
    // const int64_t signal_ndim = sizes.size() - 1;
    PADDLE_ENFORCE_EQ(signal_ndim, sizes.size() - 1,
                      platform::errors::InvalidArgument(
                          "The signal_ndim must be equal to sizes.size() - 1,"
                          "But signal_ndim is: [%d], sizes.size() - 1 is: [%d]",
                          signal_ndim, sizes.size() - 1));

    hipfftType exec_type = [&] {
      if (dtype == framework::proto::VarType::FP32) {
        switch (fft_type) {
          case FFTTransformType::C2C:
            return HIPFFT_C2C;
          case FFTTransformType::R2C:
            return HIPFFT_R2C;
          case FFTTransformType::C2R:
            return HIPFFT_C2R;
        }
      } else if (dtype == framework::proto::VarType::FP64) {
        switch (fft_type) {
          case FFTTransformType::C2C:
            return HIPFFT_Z2Z;
          case FFTTransformType::R2C:
            return HIPFFT_D2Z;
          case FFTTransformType::C2R:
            return HIPFFT_Z2D;
        }
      }
      PADDLE_THROW(platform::errors::InvalidArgument(
          "hipFFT only support transforms of type float32 and float64"));
    }();

    // disable auto allocation of workspace to use allocator from the framework
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::hipfftSetAutoAllocation(
        plan(), /* autoAllocate */ 0));

    size_t ws_size_t;

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::hipfftMakePlanMany(
        plan(), signal_ndim, signal_sizes.data(),
        /* inembed */ nullptr, /* base_istride */ 1, /* idist */ 1,
        /* onembed */ nullptr, /* base_ostride */ 1, /* odist */ 1, exec_type,
        batch, &ws_size_t));

    ws_size = ws_size_t;
  }

  const hipfftHandle& plan() const { return plan_ptr.get(); }

  FFTTransformType transform_type() const { return fft_type_; }
  ScalarType data_type() const { return value_type_; }
  size_t workspace_size() const { return ws_size; }

 private:
  HIPFFTHandle plan_ptr;
  size_t ws_size;
  FFTTransformType fft_type_;
  ScalarType value_type_;
};
#endif

// Hashing machinery for Key
// Fowler–Noll–Vo hash function
// see
// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
template <typename Key>
struct KeyHash {
  // Key must be a POD because we read out its memory
  // contenst as char* when hashing
  static_assert(std::is_pod<Key>::value, "Key must be plain old data type");

  size_t operator()(const Key& params) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&params);
    uint32_t value = 0x811C9DC5;
    for (int i = 0; i < static_cast<int>(sizeof(Key)); ++i) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return static_cast<size_t>(value);
  }
};

template <typename Key>
struct KeyEqual {
  // Key must be a POD because we read out its memory
  // contenst as char* when comparing
  static_assert(std::is_pod<Key>::value, "Key must be plain old data type");

  bool operator()(const Key& a, const Key& b) const {
    auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
    auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
    return memcmp(ptr1, ptr2, sizeof(Key)) == 0;
  }
};

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

// This cache assumes that the mapping from key to value never changes.
// This is **NOT** thread-safe. Please use a mutex when using it **AND** the
// value returned from try_emplace_value.
// The contract of using this cache is that try_emplace_value should only be
// used when the max_size is positive.
class FFTConfigCache {
 public:
  using kv_t = typename std::pair<FFTConfigKey, FFTConfig>;
  using map_t = typename std::unordered_map<
      std::reference_wrapper<FFTConfigKey>, typename std::list<kv_t>::iterator,
      KeyHash<FFTConfigKey>, KeyEqual<FFTConfigKey>>;
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
    PADDLE_ENFORCE_GT(_max_size, 0,
                      platform::errors::InvalidArgument(
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
        new_size, 0,
        platform::errors::InvalidArgument(
            "cuFFT plan cache size must be non-negative, But received is [%d]",
            new_size));
    PADDLE_ENFORCE_LE(new_size, CUFFT_MAX_PLAN_NUM,
                      platform::errors::InvalidArgument(
                          "cuFFT plan cache size can not be larger than [%d], "
                          "But received is [%d]",
                          CUFFT_MAX_PLAN_NUM, new_size));
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

}  // namespace operators
}  // namespace paddle
