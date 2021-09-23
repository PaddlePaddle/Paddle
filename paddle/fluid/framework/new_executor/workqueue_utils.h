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
#include <cassert>
#include <cstddef>
#include <cstdlib>
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || \
    defined(__i386__)
#define __WQ_x86__
#include <immintrin.h>
#endif
#include <thread>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

template <typename Holder>
class CounterGuard {
 public:
  explicit CounterGuard(Holder* holder) : counter_holder_(holder) {
    assert(holder != nullptr);
    counter_holder_->AddCounter();
  }

  ~CounterGuard() {
    if (counter_holder_ != nullptr) {
      counter_holder_->SubCounter();
    }
  }

  CounterGuard(CounterGuard&& other) : counter_holder_(other.counter_holder_) {
    other.counter_holder_ = nullptr;
  }

  CounterGuard& operator=(CounterGuard&& other) {
    counter_holder_ = other.counter_holder_;
    other.counter_holder_ = nullptr;
    return *this;
  }

  // copy constructor deleted, we define this for std::function
  // never use it directly
  CounterGuard(const CounterGuard& other) {
    PADDLE_THROW(platform::errors::Unavailable(
        "Never use the copy constructor of CounterGuard."));
  }

  CounterGuard& operator=(const CounterGuard&) = delete;

 private:
  Holder* counter_holder_{nullptr};
};

void* AlignedMalloc(size_t size, size_t alignment);

void AlignedFree(void* memory_ptr);

static inline void CpuRelax() {
#if defined(__WQ_x86__)
  _mm_pause();
#endif
}

class SpinLock {
 public:
  void lock() {
    for (;;) {
      if (!lock_.exchange(true, std::memory_order_acquire)) {
        break;
      }
      constexpr int kMaxLoop = 32;
      for (int loop = 1; lock_.load(std::memory_order_relaxed);) {
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

  void unlock() { lock_.store(false, std::memory_order_release); }

 private:
  std::atomic<bool> lock_{false};
};

}  // namespace framework
}  // namespace paddle
