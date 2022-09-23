// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <functional>
#include <thread>

namespace paddle {
namespace framework {

struct StlThreadEnvironment {
  struct Task {
    std::function<void()> f;
  };

  // EnvThread constructor must start the thread,
  // destructor must join the thread.
  class EnvThread {
   public:
    explicit EnvThread(std::function<void()> f) : thr_(std::move(f)) {}
    void WaitExit() {
      if (thr_.joinable()) {
        thr_.join();
      }
    }
    ~EnvThread() {
      if (thr_.joinable()) {
        thr_.join();
      }
    }

   private:
    std::thread thr_;
  };

  EnvThread* CreateThread(std::function<void()> f) {
    return new EnvThread(std::move(f));
  }
  Task CreateTask(std::function<void()> f) { return Task{std::move(f)}; }
  void ExecuteTask(const Task& t) { t.f(); }
};

}  // namespace framework
}  // namespace paddle
