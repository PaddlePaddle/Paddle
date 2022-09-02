/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once
#include "paddle/fluid/operators/jit/kernel_base.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {
namespace jit {

struct KernelKey {
  struct Hash {
    size_t operator()(const KernelKey& key) const {
      int place = static_cast<int>(key.place_.GetType());  // less than 2^8
      int type = static_cast<int>(key.type_) << 8;         // less than 2^(32-8)
      std::hash<int> hasher;
      return hasher(place + type);
    }
  };

  KernelType type_;
  platform::Place place_;

  KernelKey(KernelType type, platform::Place place)
      : type_(type), place_(place) {}
  size_t hash_key() const { return Hash()(*this); }

  bool operator==(const KernelKey& o) const {
    return platform::places_are_same_class(place_, o.place_) &&
           type_ == o.type_;
  }
  bool operator!=(const KernelKey& o) const { return !(*this == o); }
};

// Every JitCode should have a method to get the key from attribution
template <typename Attr>
int64_t JitCodeKey(const Attr& attr);

}  // namespace jit
}  // namespace operators
}  // namespace paddle
