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

namespace paddle {
namespace experimental {

/**
 * [ Why need Backend? ]
 *
 * Backend not only means place. Backend is a superset of place.
 *
 * Place cannot indicate the difference in calculation methods on the device,
 * but in order to make the boundary of the kernel clearer and the function
 * more specific, we need to distinguish the calculation method.
 *
 * Such as the kernel for CUDA device, it can be a native CUDA kernel,
 * or a kernel implemented by CUDNN library.
 *
 * Note(chenweihang): HIP is not needed now, we can added it if needed
 * in the future
 */
enum class Backend : uint8_t {
  // kernel backend cannot be undefined
  UNDEFINED = 0,

  // basic kernel backend
  CPU,

  // various acceleration devices' backends
  CUDA,
  XPU,  // XPU currently does not exist at the same time as CUDA
  NPU,  // NPU currently does not exist at the same time as CUDA

  // the third library backend
  MKLDNN,
  CUDNN,

  // end of backend types
  kNumBackends,
};

/**
 * We use the backend to form a bit set to assist the runtime kernel selection,
 * and the higher backend bit has a higher priority.
 *
 * A Tensor may belong to multiple backends at the same time, such CUDNN and
 * CUDA. Only one backend value cannot
 */
class BackendSet final {
 public:
  constexpr BackendSet() : bitset_(0) {}
  explicit constexpr BackendSet(Backend b)
      : bitset_(b == Backend::UNDEFINED ? 0 : 1ULL << (static_cast<uint8_t>(b) -
                                                       1)) {}

  uint64_t bitset() const { return bitset_; }

  bool inline Has(Backend b) const {
    // TODO(chenweihang): replace by internal assert method later
    if (b == Backend::UNDEFINED) {
      throw std::runtime_error("Backend argument can't be UNDEFINED.");
    }
    return static_cast<bool>(bitset_ & BackendSet(b).bitset())
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
  constexpr BackendSet(uint64_t bitset) : bitset_(bitset) {}
  uint64_t bitset_;
};

std::ostream& operator<<(std::ostream& os, Backend backend) {
  switch (backend) {
    case Backend::UNDEFINED:
      os << "Undefined";
      break;
    case Backend::CPU:
      os << "CPU";
      break;
    case Backend::CUDA:
      os << "CUDA";
      break;
    case Backend::XPU:
      os << "XPU";
      break;
    case Backend::NPU:
      os << "NPU";
      break;
    case Backend::MKLDNN:
      os << "MKLDNN";
      break;
    case Backend::CUDNN:
      os << "CUDNN";
      break;
    default:
      // TODO(chenweihang): replace by internal enforce method later
      throw std::runtime_error("Invalid Backend type.");
  }
  return os;
}

}  // namespace experimental
}  // namespace paddle
