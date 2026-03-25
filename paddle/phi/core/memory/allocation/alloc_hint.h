// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

namespace paddle {
namespace memory {
namespace allocation {

enum class PoolHint : uint8_t {
  kNone = 0,
  kStable = 1,
  kLongLived = 2,
};

inline PoolHint& CurrentPoolHintStorage() {
  static thread_local PoolHint hint = PoolHint::kNone;
  return hint;
}

inline PoolHint GetCurrentPoolHint() { return CurrentPoolHintStorage(); }

inline void SetCurrentPoolHint(PoolHint hint) {
  CurrentPoolHintStorage() = hint;
}

class PoolHintGuard {
 public:
  explicit PoolHintGuard(PoolHint hint) : previous_hint_(GetCurrentPoolHint()) {
    SetCurrentPoolHint(hint);
  }

  ~PoolHintGuard() { SetCurrentPoolHint(previous_hint_); }

  PoolHintGuard(const PoolHintGuard&) = delete;
  PoolHintGuard& operator=(const PoolHintGuard&) = delete;

 private:
  PoolHint previous_hint_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
