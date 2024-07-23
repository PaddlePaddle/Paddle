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

#include "paddle/fluid/distributed/fleet_executor/task_loop_thread.h"

#include "paddle/common/errors.h"
#include "paddle/fluid/distributed/fleet_executor/task_loop.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::distributed {

TaskLoopThread::TaskLoopThread() : start_(false), loop_(nullptr) {}

TaskLoopThread::~TaskLoopThread() {
  if (loop_ != nullptr) {
    loop_->Quit();
    thread_.join();
  }
}

TaskLoop* TaskLoopThread::StartLoop() {
  PADDLE_ENFORCE_EQ(
      start_,
      false,
      phi::errors::PreconditionNotMet("thread is already running."));
  start_ = true;
  thread_ = std::thread([this]() { Loop(); });

  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [=] { return loop_ != nullptr; });
  return loop_;
}

void TaskLoopThread::Loop() {
  TaskLoop loop;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    loop_ = &loop;
    cv_.notify_one();
  }
  loop.Loop();

  std::unique_lock<std::mutex> lock(mutex_);
  loop_ = nullptr;
}

}  // namespace paddle::distributed
