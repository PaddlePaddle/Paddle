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
#if !defined(_WIN32)
#include <sched.h>
#else
#include <windows.h>
#endif  // !_WIN32

#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace memory {

class SpinLock {
 public:
  SpinLock() : mlock_(false) {}

  void lock() {
    bool expect = false;
    uint64_t spin_cnt = 0;
    while (!mlock_.compare_exchange_weak(expect, true)) {
      expect = false;
      if ((++spin_cnt & 0xFF) == 0) {
#if defined(_WIN32)
        SleepEx(50, FALSE);
#else
        sched_yield();
#endif
      }
    }
  }

  void unlock() { mlock_.store(false); }
  DISABLE_COPY_AND_ASSIGN(SpinLock);

 private:
  std::atomic<bool> mlock_;
};

}  // namespace memory
}  // namespace paddle
