/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <ostream>

#include "paddle/common/exception.h"
#include "paddle/phi/common/backend.h"
namespace paddle {
namespace experimental {

/**
 * We use the backend to form a bit set to assist the runtime kernel selection,
 * and the higher backend bit has a higher priority.
 *
 * A Tensor may belong to multiple backends at the same time, such CPU and
 * OneDNN. Only one backend value cannot
 */
class BackendSet final {
 public:
  constexpr BackendSet() : bitset_(0) {}
  explicit constexpr BackendSet(Backend b)
      : bitset_(b == Backend::UNDEFINED
                    ? 0
                    : 1ULL << (static_cast<uint8_t>(b) - 1)) {}

  inline uint32_t bitset() const { return bitset_; }

  bool inline Has(Backend b) const {
    PD_CHECK(b != Backend::UNDEFINED, "Backend argument can't be UNDEFINED.");
    return static_cast<bool>(bitset_ & BackendSet(b).bitset());
  }
  bool IsEmpty() const { return bitset_ == 0; }

  BackendSet operator|(const BackendSet& other) const {
    return BackendSet(bitset_ | other.bitset());
  }
  BackendSet operator&(const BackendSet& other) const {
    return BackendSet(bitset_ & other.bitset());
  }
  BackendSet operator-(const BackendSet& other) const {
    return BackendSet(bitset_ & ~other.bitset());
  }
  BackendSet operator^(const BackendSet& other) const {
    return BackendSet(bitset_ ^ other.bitset());
  }

  bool operator==(const BackendSet& other) const {
    return bitset_ == other.bitset();
  }

 private:
  constexpr BackendSet(uint32_t bitset) : bitset_(bitset) {}
  uint32_t bitset_;
};

}  // namespace experimental
}  // namespace paddle
