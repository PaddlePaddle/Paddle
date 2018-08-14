// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <stdint.h>
#include <stdlib.h>
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace platform {
namespace random {

template <size_t N>
struct LargeInt {
 public:
  HOSTDEVICE inline LargeInt() {
    for (size_t i = 0; i < N; ++i) {
      data_[i] = 0;
    }
  }

  HOSTDEVICE inline const uint32_t& operator[](int index) const {
    return data_[index];
  }

  HOSTDEVICE inline uint32_t& operator[](int index) { return data_[index]; }

  constexpr static size_t size = N;

  HOSTDEVICE LargeInt<N>& operator=(uint64_t val) {
    data_[0] = static_cast<uint32_t>(val);
    data_[1] = static_cast<uint32_t>(val >> 32);
    return *this;
  }

  HOSTDEVICE LargeInt<N>& operator+=(uint64_t val) {
    auto low = static_cast<uint32_t>(val);
    auto hi = static_cast<uint32_t>(val >> 32);
    data_[0] += low;
    if (data_[0] < low) {  // overflow
      ++hi;
    }
    data_[1] += hi;
    if (data_[1] < hi) {      // overflow
      if (++data_[2] == 0) {  // overflow
        ++data_[3];
      }
    }
    return *this;
  }

  HOSTDEVICE LargeInt<N>& operator++() {
    for (size_t i = 0; i < N; ++i) {
      if (++data_[i] != 0) {  // overflow
        break;
      }
    }
    return *this;
  }

 private:
  uint32_t data_[N];
};

struct Philox32x4State {
  LargeInt<2> key_;
  LargeInt<4> counter_;

  inline HOSTDEVICE Philox32x4State() {}
  inline HOSTDEVICE Philox32x4State(uint64_t seed) { key_ = seed; }
};

class Philox32x4 {
 public:
  using ResultElementType = uint32_t;
  using ResultType = LargeInt<4>;
  using Key = LargeInt<2>;
  constexpr static size_t N = LargeInt<4>::size;

  HOSTDEVICE inline Philox32x4() {}

  HOSTDEVICE inline explicit Philox32x4(uint64_t seed) : state_(seed) {}

  HOSTDEVICE ResultType operator()() {
    ResultType counter = state_.counter_;
    Key key = state_.key_;

    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    ++state_.counter_;
    return counter;
  }

  HOSTDEVICE void Discard(uint64_t value) { state_.counter_ += value; }

  HOSTDEVICE Philox32x4State State() const { return state_; }

 private:
  static constexpr uint32_t kPhiloxW32A = 0x9E3779B9;
  static constexpr uint32_t kPhiloxW32B = 0xBB67AE85;
  static constexpr uint32_t kPhiloxM4x32A = 0xD2511F53;
  static constexpr uint32_t kPhiloxM4x32B = 0xCD9E8D57;

  // Helper function for a single round of the underlying Philox algorithm.
  HOSTDEVICE static ResultType ComputeSingleRound(const ResultType& counter,
                                                  const Key& key) {
    uint32_t lo0;
    uint32_t hi0;
    MultiplyHighLow(kPhiloxM4x32A, counter[0], &lo0, &hi0);

    uint32_t lo1;
    uint32_t hi1;
    MultiplyHighLow(kPhiloxM4x32B, counter[2], &lo1, &hi1);

    ResultType result;
    result[0] = hi1 ^ counter[1] ^ key[0];
    result[1] = lo1;
    result[2] = hi0 ^ counter[3] ^ key[1];
    result[3] = lo0;
    return result;
  }

  // Helper function to return the lower and higher 32-bits from two 32-bit
  // integer multiplications.
  HOSTDEVICE
  inline static void MultiplyHighLow(uint32_t a, uint32_t b,
                                     uint32_t* result_low,
                                     uint32_t* result_high) {
#ifndef __CUDA_ARCH__
    const uint64_t product = static_cast<uint64_t>(a) * b;
    *result_low = static_cast<uint32_t>(product);
    *result_high = static_cast<uint32_t>(product >> 32);
#else
    *result_low = a * b;
    *result_high = __umulhi(a, b);
#endif
  }

  HOSTDEVICE inline static void RaiseKey(Key* key) {
    (*key)[0] += kPhiloxW32A;
    (*key)[1] += kPhiloxW32B;
  }

 private:
  Philox32x4State state_;
};

}  // namespace random
}  // namespace platform
}  // namespace paddle
