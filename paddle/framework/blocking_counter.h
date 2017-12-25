/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <mutex>
#include <thread>

#include "paddle/platform/call_once.h"
#include "paddle/platform/enforce.h"

namespace paddle {
namespace framework {

class BlockingCounter {
 public:
  BlockingCounter(int cnt) : cnt_(cnt), done_(false) {
    PADDLE_ENFORCE_GE(static_cast<int>(cnt_), 0,
                      "The initialized counter should not be less than 0");
  }

  ~BlockingCounter() {}

  /**
   * @breif   Wait untile the count was decreased to 0
   */
  void Wait() {
    if (cnt_ == 0) return;
    std::unique_lock<std::mutex> lock(done_m);
    done_cv.wait(lock, [=] { return done_ == true; });
  }

  /**
   * @breif   Descrease the count by 1
   */
  void DecreaseCount() {
    cnt_.fetch_sub(1);
    if (cnt_ != 0) return;
    std::unique_lock<std::mutex> lock(done_m);
    done_ = true;
    lock.unlock();
    done_cv.notify_all();
  }

 private:
  BlockingCounter(const BlockingCounter&) = delete;
  std::atomic<int> cnt_;
  bool done_;
  std::mutex done_m;
  std::condition_variable done_cv;
};

}  // namespace framework
}  // namespace paddle
