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

#include "paddle/fluid/distributed/fleet_executor/task_loop.h"

#include "paddle/common/errors.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::distributed {

thread_local TaskLoop* TaskLoop::thread_local_loop_ = nullptr;

TaskLoop* TaskLoop::GetTaskLoopOfCurrentThread() { return thread_local_loop_; }

TaskLoop::TaskLoop()
    : looping_(false), quit_(false), thread_id_(std::this_thread::get_id()) {
  PADDLE_ENFORCE_EQ(
      thread_local_loop_,
      nullptr,
      phi::errors::AlreadyExists("Another TaskLoop is already init."));
  thread_local_loop_ = this;
}

TaskLoop::~TaskLoop() { thread_local_loop_ = nullptr; }

void TaskLoop::Loop() {
  PADDLE_ENFORCE_EQ(looping_,
                    false,
                    phi::errors::PreconditionNotMet(
                        "Loop can only execute in one loop thread"));
  AssertInLoopThread();

  looping_ = true;
  quit_ = false;

  while (!quit_) {
    auto tasks = tasks_.PopAll();
    for (auto& task : tasks) {
      task();
    }
  }
  looping_ = false;
}

void TaskLoop::Quit() {
  quit_ = true;
  if (!IsInLoopThread()) WakeUp();
}

void TaskLoop::RunInLoop(Functor cb) {
  if (IsInLoopThread()) {
    cb();
  } else {
    QueueInLoop(cb);
  }
}

void TaskLoop::QueueInLoop(Functor cb) { tasks_.Push(cb); }

void TaskLoop::WakeUp() {
  Functor task([] {});
  QueueInLoop(task);
}

void TaskLoop::AbortNotInLoopThread() {
  PADDLE_THROW(phi::errors::PreconditionNotMet(
      "This TaskLoop was created in thread %d, but current thread is %d",
      thread_id_,
      std::this_thread::get_id()));
}

}  // namespace paddle::distributed
