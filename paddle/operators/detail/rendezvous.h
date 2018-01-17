//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>

namespace paddle {
namespace operators {
namespace detail {

template <typename Item>
class SimpleBlockQueue {
 public:
  void Push(Item const& value) {
    {
      std::unique_lock<std::mutex> lock(this->mutex_);
      queue_.push_front(value);
    }
    this->condition_.notify_one();
  }

  Item Pop() {
    std::unique_lock<std::mutex> lock(this->mutex_);
    this->condition_.wait(lock, [=] { return !this->queue_.empty(); });
    Item rc(std::move(this->queue_.back()));
    this->queue_.pop_back();
    return rc;
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<Item> queue_;
};

}  // namespace detail
}  // namespace operators
}  // namespace paddle
