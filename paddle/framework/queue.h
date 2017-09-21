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
#include <deque>
#include <functional>
#include <mutex>

#include "paddle/platform/enforce.h"

namespace paddle {
namespace framework {

template <typename Item>
struct QueueItem {
  using RecvCallback = std::function<void(const Item&)>;
  Item item_;
  RecvCallback recv_callback_;

  bool ReadyForRecv() const { return recv_callback_ != nullptr; }

  // enable implicit cast from QueueItem to Item.
  operator const Item&() const { return item_; }
};

template <typename Item>
class Queue {
 public:
  using QItem = QueueItem<Item>;
  using RecvCallback = typename QueueItem<Item>::RecvCallback;

  void Push(Item item) {
    std::lock_guard<std::mutex> guard(mu_);

    // Recv has been invoked.
    if (!queue_.empty() && queue_.front()->ReadyForRecv()) {
      auto& qitem = *queue_.front();
      qitem.recv_callback_(item);
      queue_.pop_front();
      return;
    }

    queue_.emplace_back(new QItem{
        item,   /* item_ */
        nullptr /* recv_callback_ */
    });
  }

  void PullAsync(RecvCallback on_complete) {
    PADDLE_ENFORCE(on_complete != nullptr);

    std::lock_guard<std::mutex> guard(mu_);

    // Have not sent
    if (queue_.empty() || queue_.front()->ReadyForRecv()) {
      auto item = new QItem();
      item->recv_callback_ = on_complete;
      queue_.emplace_back(item);
      return;
    }

    auto& item = *queue_.front();
    on_complete(item);
    queue_.pop_front();
  }

  // A simple wrapper for RecvAsync
  Item Pull() {
    std::condition_variable cv;
    std::mutex cv_mtx;
    bool done = false;
    Item ret_val;

    PullAsync([&ret_val, &cv_mtx, &done, &cv](const Item& o) {
      ret_val = o;
      {
        std::lock_guard<std::mutex> g(cv_mtx);
        done = true;
      }
      cv.notify_one();
    });

    std::unique_lock<std::mutex> lk(cv_mtx);
    cv.wait(lk, [&done] { return done; });
    return ret_val;
  }

 private:
  std::deque<std::unique_ptr<QItem>> queue_;
  std::mutex mu_;
};

}  // namespace framework
}  // namespace paddle
