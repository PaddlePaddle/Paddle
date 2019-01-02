// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/distributed/barrier.h"

namespace paddle {
namespace operators {
namespace distributed {

void Barrier::Wait() {
  std::unique_lock<std::mutex> lock(this->mutex_);
  barrier_cond_.wait(
      lock, [this] { return counter_ == worker_size_ && worker_size_ != 0; });
}

void Barrier::Notify() { barrier_cond_.notify_all(); }

void Barrier::Increase() {
  std::unique_lock<std::mutex> lock(this->mutex_);
  ++counter_;
  if (counter_ >= worker_size_) {
    lock.unlock();
    barrier_cond_.notify_all();
    lock.lock();
  }
}

void Barrier::Decrease() {
  std::unique_lock<std::mutex> lock(this->mutex_);
  --counter_;
  if (counter_ >= worker_size_) {
    lock.unlock();
    barrier_cond_.notify_all();
    lock.lock();
  }
}

void Barrier::Reset() {
  std::unique_lock<std::mutex> lock(this->mutex_);
  counter_ = 0;
}

void Barrier::SetWorkerSize(int worker_size) {
  std::unique_lock<std::mutex> lock(this->mutex_);
  worker_size_ = worker_size;
  lock.unlock();
  barrier_cond_.notify_all();
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
