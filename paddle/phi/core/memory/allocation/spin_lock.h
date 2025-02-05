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

#include <atomic>
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || \
    defined(__i386__)
#define __PADDLE_x86__
#include <immintrin.h>
#endif
#include <thread>

#include "paddle/common/macros.h"

namespace paddle {
namespace memory {
static inline void CpuRelax() {
#if defined(__PADDLE_x86__)
  _mm_pause();
#endif
}

class SpinLock {
 public:
  SpinLock() : mlock_(false) {}

  void lock() {
    for (;;) {
      if (!mlock_.exchange(true, std::memory_order_acquire)) {
        break;
      }
      constexpr int kMaxLoop = 32;
      for (int loop = 1; mlock_.load(std::memory_order_relaxed);) {
        if (loop <= kMaxLoop) {
          for (int i = 1; i <= loop; ++i) {
            CpuRelax();
          }
          loop *= 2;
        } else {
          std::this_thread::yield();
        }
      }
    }
  }

  void unlock() { mlock_.store(false, std::memory_order_release); }

  DISABLE_COPY_AND_ASSIGN(SpinLock);

 private:
  std::atomic<bool> mlock_;
};

}  // namespace memory
}  // namespace paddle
