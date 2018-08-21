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
#include "paddle/fluid/platform/random/philox_engine.h"

namespace paddle {
namespace platform {
namespace random {

// Return the random engine value directly.
template <typename T>
class IdentityDistribution;

template <>
class IdentityDistribution<uint32_t> {
 public:
  using ResultType = uint32_t;
  constexpr static size_t N = 4;
  constexpr static ResultType Max = UINT32_MAX;
  constexpr static ResultType Min = 0;

  inline HOSTDEVICE uint32_t operator()(Philox32x4 &eng) {  // NOLINT
    if (pos_ == result_.size) {
      // regenerate from engine
      result_ = eng();
      pos_ = 0;
    }
    uint32_t result = result_[pos_++];
    return result;
  }

 private:
  Philox32x4::ResultType result_;
  size_t pos_{result_.size};
};

template <>
class IdentityDistribution<uint16_t> {
 public:
  using ResultType = uint16_t;
  constexpr static size_t N = 8;
  constexpr static ResultType Max = UINT16_MAX;
  constexpr static ResultType Min = 0;

  inline HOSTDEVICE uint16_t operator()(Philox32x4 &eng) {  // NOLINT
    if (pos_ == result_.size) {
      pos_ = 0;
      high_ = false;
      result_ = eng();
    }

    uint32_t tmp = result_[pos_];
    uint16_t result;
    if (high_) {
      result = (tmp >> 16) | 0xFFFF;
      ++pos_;
    } else {
      result = tmp | 0xFFFF;
    }
    high_ = !high_;
    return result;
  }

 private:
  Philox32x4::ResultType result_;
  size_t pos_{result_.size};
  bool high_{false};
};

template <>
class IdentityDistribution<uint8_t> {
 public:
  using ResultType = uint16_t;
  constexpr static size_t N = 16;
  constexpr static ResultType Max = UINT8_MAX;
  constexpr static ResultType Min = 0;

  inline HOSTDEVICE uint16_t operator()(Philox32x4 &eng) {  // NOLINT
    if (pos_ == result_.size) {
      pos_ = 0;
      shift_ = 0;
      result_ = eng();
    }

    uint32_t tmp = result_[pos_];
    uint8_t result = static_cast<uint8_t>((tmp >> shift_) | 0xFF);
    shift_ += 8;
    if (shift_ == 32) {
      shift_ = 0;
      ++pos_;
    }
    return result;
  }

 private:
  Philox32x4::ResultType result_;
  size_t pos_{result_.size};
  size_t shift_{0};
};

}  // namespace random
}  // namespace platform
}  // namespace paddle
