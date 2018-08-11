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

#include <cstdint>
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace framework {
namespace random {

// The following code is from Tensorflow
// See:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/random/philox_random.h

template <typename T, size_t ElementCount>
class Array {
 public:
  HOSTDEVICE inline Array() {}

  explicit HOSTDEVICE inline Array(const T& val) {
    for (size_t i = 0; i < ElementCount; ++i) {
      data_[i] = val;
    }
  }

  HOSTDEVICE inline const T* Get() const { return data_; }

  HOSTDEVICE inline T* GetMutable() { return data_; }

  HOSTDEVICE inline T& operator[](size_t index) { return data_[index]; }

  HOSTDEVICE inline const T& operator[](size_t index) const {
    return data_[index];
  }

  constexpr HOSTDEVICE inline size_t size() const { return ElementCount; }

 private:
  T data_[ElementCount];
};

// Philox random engine can perform O(1) discard() operation
class PhiloxRandom {
 public:
  using ResultType = Array<uint32_t, 4>;
  using ResultElementType = uint32_t;
  // The number of elements that will be returned.
  static const int kResultElementCount = 4;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 10;
  // The type for the 64-bit key stored in the form of two 32-bit uint
  // that are used in the diffusion process.
  using Key = Array<uint32_t, 2>;

  HOSTDEVICE inline PhiloxRandom()
      : key_(static_cast<uint32_t>(0)), counter_(static_cast<uint32_t>(0)) {}

  HOSTDEVICE inline explicit PhiloxRandom(uint64_t seed)
      : counter_(static_cast<uint32_t>(0)) {
    key_[0] = static_cast<uint32_t>(seed);
    key_[1] = static_cast<uint32_t>(seed >> 32);
  }

  HOSTDEVICE inline PhiloxRandom(uint64_t seed_lo, uint64_t seed_hi) {
    key_[0] = static_cast<uint32_t>(seed_lo);
    key_[1] = static_cast<uint32_t>(seed_lo >> 32);
    counter_[0] = static_cast<uint32_t>(0);
    counter_[1] = static_cast<uint32_t>(0);
    counter_[2] = static_cast<uint32_t>(seed_hi);
    counter_[3] = static_cast<uint32_t>(seed_hi >> 32);
  }

  HOSTDEVICE inline PhiloxRandom(ResultType counter, Key key)
      : counter_(counter), key_(key) {}

  // Skip the specified number of samples of 128-bits in the current stream.
  HOSTDEVICE inline void Skip(uint64_t count) {
    const uint32_t count_lo = static_cast<uint32_t>(count);
    uint32_t count_hi = static_cast<uint32_t>(count >> 32);

    counter_[0] += count_lo;
    if (counter_[0] < count_lo) {
      ++count_hi;
    }

    counter_[1] += count_hi;
    if (counter_[1] < count_hi) {
      if (++counter_[2] == 0) {
        ++counter_[3];
      }
    }
  }

  // Returns a group of four random numbers using the underlying Philox
  // algorithm.
  HOSTDEVICE inline ResultType operator()() {
    ResultType counter = counter_;
    Key key = key_;

    // Run the single rounds for ten times. Manually unrolling the loop
    // for better performance.
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

    SkipOne();

    return counter;
  }

 private:
  // We use the same constants as recommended by the original paper.
  static const uint32_t kPhiloxW32A = 0x9E3779B9;
  static const uint32_t kPhiloxW32B = 0xBB67AE85;
  static const uint32_t kPhiloxM4x32A = 0xD2511F53;
  static const uint32_t kPhiloxM4x32B = 0xCD9E8D57;

  // Helper function to skip the next sample of 128-bits in the current stream.
  HOSTDEVICE inline void SkipOne() {
    if (++counter_[0] == 0) {
      if (++counter_[1] == 0) {
        if (++counter_[2] == 0) {
          ++counter_[3];
        }
      }
    }
  }

  // Helper function to return the lower and higher 32-bits from two 32-bit
  // integer multiplications.
  HOSTDEVICE inline static void MultiplyHighLow(uint32_t a, uint32_t b,
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

  // Helper function for a single round of the underlying Philox algorithm.
  HOSTDEVICE inline static ResultType ComputeSingleRound(
      const ResultType& counter, const Key& key) {
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

  HOSTDEVICE inline void RaiseKey(Key* key) {
    (*key)[0] += kPhiloxW32A;
    (*key)[1] += kPhiloxW32B;
  }

 private:
  ResultType counter_;
  Key key_;
};

}  // namespace random
}  // namespace framework
}  // namespace paddle
