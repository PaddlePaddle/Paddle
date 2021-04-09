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

#include "paddle/fluid/imperative/tracer.h"
#ifndef _WIN32
#include <windows.h>
#endif

namespace paddle {
namespace pybind {
#define VARBASEPOOLSIZE 500
class VarBasePool {
 public:
  static inline VarBasePool& GetInstance() {
    static VarBasePool instance;
    return instance;
  }

  VarBasePool() {
    running_ = true;
    auto tracer = imperative::GetCurrentTracer();
    for (int i = 0; i < VARBASEPOOLSIZE; i++) {
      pool_[i] = std::shared_ptr<imperative::VarBase>(
          new imperative::VarBase(tracer->GenerateUniqueName()));
    }
    head_ = VARBASEPOOLSIZE - 1;
    tail_ = 0;
    thread_ = new std::thread(&VarBasePool::thread_loop, this);
  }

  ~VarBasePool() {
    running_ = false;
    thread_->join();
  }

  inline std::shared_ptr<imperative::VarBase> Get() {
    while (pool_[tail_] == nullptr) {
      tail_ = (tail_ + 1) % VARBASEPOOLSIZE;
    }
    auto ret = pool_[tail_];
    pool_[tail_] = nullptr;
    tail_ = (tail_ + 1) % VARBASEPOOLSIZE;
    return ret;
  }

 private:
  std::shared_ptr<imperative::VarBase> pool_[VARBASEPOOLSIZE];
  int head_;
  int tail_;
  std::thread* thread_;
  bool running_;

  void thread_loop() {
    while (1) {
      if (!running_) {
        return;
      }
      while (pool_[(head_ + 1) % VARBASEPOOLSIZE] == nullptr && running_) {
        auto tracer = imperative::GetCurrentTracer();
        if (tracer) {
          head_ = (head_ + 1) % VARBASEPOOLSIZE;
          pool_[head_] = std::shared_ptr<imperative::VarBase>(
              new imperative::VarBase(tracer->GenerateUniqueName()));
        }
      }
#ifndef _WIN32
      Sleep(1);
#else
      usleep(1000);
#endif
    }
  }
};
}  // namespace pybind
}  // namespace paddle
