// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <assert.h>

namespace paddle {
namespace framework {

class Notification {
 public:
  Notification() : notified_(0) {}
  ~Notification() {
    // In case the notification is being used to synchronize its own deletion,
    // force any prior notifier to leave its critical section before the object
    // is destroyed.
    const std::lock_guard<std::mutex> l(mu_);
  }

  void Notify() {
    const std::lock_guard<std::mutex> l(mu_);
    assert(!HasBeenNotified());
    notified_.store(true, std::memory_order_release);
    cv_.notify_all();
  }

  bool HasBeenNotified() const {
    return notified_.load(std::memory_order_acquire);
  }

  void WaitForNotification() {
    if (!HasBeenNotified()) {
      std::unique_lock<std::mutex> l(mu_);
      while (!HasBeenNotified()) {
        cv_.wait(l);
      }
    }
  }

 private:
  friend bool WaitForNotificationWithTimeout(Notification* n,
                                             int64_t timeout_in_us);
  bool WaitForNotificationWithTimeout(int64_t timeout_in_us) {
    bool notified = HasBeenNotified();
    if (!notified) {
      std::unique_lock<std::mutex> l(mu_);
      do {
        notified = HasBeenNotified();
      } while (!notified &&
               cv_.wait_for(l, std::chrono::microseconds(timeout_in_us)) !=
                   std::cv_status::timeout);
    }
    return notified;
  }

  std::mutex mu_;               // protects mutations of notified_
  std::condition_variable cv_;  // signaled when notified_ becomes non-zero
  std::atomic<bool> notified_;  // mutations under mu_
};

inline bool WaitForNotificationWithTimeout(Notification* n,
                                           int64_t timeout_in_us) {
  return n->WaitForNotificationWithTimeout(timeout_in_us);
}
}
}
