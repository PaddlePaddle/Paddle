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

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <type_traits>

#include "paddle/phi/common/data_type.h"

namespace phi {
namespace funcs {

// ---------------------------------------------------------------------------
// A 32-bit MT19937 engine that is bit-for-bit identical to PyTorch's
// at::mt19937 (aten/src/ATen/core/MT19937RNGEngine.h), including the seeding
// algorithm (only the lower 32 bits of the seed are used) and the byte order
// of random64() (the first draw fills the high 32 bits). The same algorithm
// is also used by cpu/randperm_kernel.cc for torch-compatible results.
// ---------------------------------------------------------------------------
class TorchMT19937Engine {
 public:
  static constexpr int kStateN = 624;
  static constexpr int kStateM = 397;
  static constexpr uint32_t kMatrixA = 0x9908b0df;
  static constexpr uint32_t kUMask = 0x80000000;
  static constexpr uint32_t kLMask = 0x7fffffff;

  inline explicit TorchMT19937Engine(uint64_t seed = 5489) {
    init_with_uint32(seed);
  }

  inline uint32_t operator()() {
    if (--(left_) == 0) {
      next_state();
    }
    uint32_t y = *(state_.data() + next_++);
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);
    return y;
  }

  // Matches at::CPUGeneratorImpl::random64(): the first 32-bit draw goes to
  // the high bits.
  inline uint64_t random64() {
    uint32_t r1 = (*this)();
    uint32_t r2 = (*this)();
    return (static_cast<uint64_t>(r1) << 32) | static_cast<uint64_t>(r2);
  }

 private:
  std::array<uint32_t, kStateN> state_;
  int left_ = 1;
  uint32_t next_ = 0;

  inline void init_with_uint32(uint64_t seed) {
    state_[0] = seed & 0xffffffff;
    for (int j = 1; j < kStateN; j++) {
      state_[j] = (1812433253 * (state_[j - 1] ^ (state_[j - 1] >> 30)) + j);
    }
    left_ = 1;
    next_ = 0;
  }

  inline uint32_t mix_bits(uint32_t u, uint32_t v) {
    return (u & kUMask) | (v & kLMask);
  }

  inline uint32_t twist(uint32_t u, uint32_t v) {
    return (mix_bits(u, v) >> 1) ^ (v & 1 ? kMatrixA : 0);
  }

  inline void next_state() {
    uint32_t *p = state_.data();
    left_ = kStateN;
    next_ = 0;

    for (int j = kStateN - kStateM + 1; --j; p++) {
      *p = p[kStateM] ^ twist(p[0], p[1]);
    }
    for (int j = kStateM; --j; p++) {
      *p = p[kStateM - kStateN] ^ twist(p[0], p[1]);
    }
    *p = p[kStateM - kStateN] ^ twist(p[0], state_[0]);
  }
};

// Bit-for-bit replication of at::transformation::uniform_real<float>
// (aten/src/ATen/core/TransformationHelper.h): take the low 24 bits of one
// 32-bit draw as the mantissa and scale into [from, to).
inline float TorchUniformReal(TorchMT19937Engine *engine,
                              float from,
                              float to) {
  constexpr uint32_t kMask = (static_cast<uint32_t>(1) << 24) - 1;
  constexpr float kDivisor = 1.0f / (static_cast<uint32_t>(1) << 24);
  float x = static_cast<float>((*engine)() & kMask) * kDivisor;
  return x * (to - from) + from;
}

// Bit-for-bit replication of at::transformation::uniform_real<double>: take
// the low 53 bits of one 64-bit draw (two 32-bit draws, high bits first).
inline double TorchUniformReal(TorchMT19937Engine *engine,
                               double from,
                               double to) {
  constexpr uint64_t kMask = (static_cast<uint64_t>(1) << 53) - 1;
  constexpr double kDivisor = 1.0 / (static_cast<uint64_t>(1) << 53);
  double x = static_cast<double>(engine->random64() & kMask) * kDivisor;
  return x * (to - from) + from;
}

// Fills `data` with samples that are bit-for-bit identical to PyTorch's CPU
// `Tensor.uniform_(min, max)` (uniform_kernel in
// aten/src/ATen/native/cpu/DistributionTemplates.h), given an engine in the
// same state as the torch generator. float/float16/bfloat16 compute in float
// (torch's opmath_t) and consume one 32-bit draw per element; double consumes
// two 32-bit draws per element. Like torch, a value that lands exactly on
// `max` after the cast to T is clamped back to `min` to keep [min, max).
template <typename T>
inline void UniformRealDistributionTorchAligned(T *data,
                                                const int64_t size,
                                                const double min,
                                                const double max,
                                                TorchMT19937Engine *engine) {
  using ComputeT = std::conditional_t<std::is_same_v<T, double>, double, float>;
  const ComputeT from = static_cast<ComputeT>(min);
  const ComputeT to = static_cast<ComputeT>(max);
  const T from_scalar = static_cast<T>(min);
  const T to_scalar = static_cast<T>(max);
  for (int64_t i = 0; i < size; ++i) {
    T value = static_cast<T>(TorchUniformReal(engine, from, to));
    data[i] = (value == to_scalar) ? from_scalar : value;
  }
}

}  // namespace funcs

template <typename T>
inline void UniformRealDistribution(T *data,
                                    const int64_t &size,
                                    const float &min,
                                    const float &max,
                                    std::shared_ptr<std::mt19937_64> engine) {
  std::uniform_real_distribution<T> dist(static_cast<T>(min),
                                         static_cast<T>(max));
  for (int64_t i = 0; i < size; ++i) {
    data[i] = dist(*engine);
  }
}

template <>
inline void UniformRealDistribution(phi::bfloat16 *data,
                                    const int64_t &size,
                                    const float &min,
                                    const float &max,
                                    std::shared_ptr<std::mt19937_64> engine) {
  std::uniform_real_distribution<float> dist(min, max);
  for (int64_t i = 0; i < size; ++i) {
    data[i] = static_cast<phi::bfloat16>(dist(*engine));
  }
}

template <>
inline void UniformRealDistribution(phi::float16 *data,
                                    const int64_t &size,
                                    const float &min,
                                    const float &max,
                                    std::shared_ptr<std::mt19937_64> engine) {
  std::uniform_real_distribution<float> dist(min, max);
  for (int64_t i = 0; i < size; ++i) {
    data[i] = static_cast<phi::float16>(dist(*engine));
  }
}

}  // namespace phi
