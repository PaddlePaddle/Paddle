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

#include "paddle/fluid/distributed/fleet_executor/task_loop_thread_pool.h"

#include "paddle/fluid/distributed/fleet_executor/task_loop.h"
#include "paddle/fluid/distributed/fleet_executor/task_loop_thread.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace distributed {

TaskLoopThreadPool::TaskLoopThreadPool() : TaskLoopThreadPool(1) {}

TaskLoopThreadPool::TaskLoopThreadPool(int thread_num)
    : start_(false), thread_num_(thread_num) {}

TaskLoopThreadPool::~TaskLoopThreadPool() = default;

void TaskLoopThreadPool::Start() {
  PADDLE_ENFORCE_EQ(start_, false, platform::errors::PreconditionNotMet(
                                       "thread pool is already start."));
  PADDLE_ENFORCE_GT(
      thread_num_, 0,
      platform::errors::InvalidArgument(
          "thread num must greater than 0, but now is %d", thread_num_));

  start_ = true;
  for (int i = 0; i < thread_num_; ++i) {
    threads_.emplace_back(new TaskLoopThread());
    loops_.push_back(threads_[i]->StartLoop());
  }
}

TaskLoop* TaskLoopThreadPool::GetLoop(int tid) {
  PADDLE_ENFORCE_EQ(start_, true, platform::errors::PreconditionNotMet(
                                      "thread pool must start first."));
  PADDLE_ENFORCE_GE(tid, 0, platform::errors::OutOfRange(
                                "tid must >= 0, but now is %d", tid));
  PADDLE_ENFORCE_LT(tid, thread_num_,
                    platform::errors::OutOfRange(
                        "tid must < thread_num, but now tid=%d thread_num=%d",
                        tid, thread_num_));
  return loops_[tid];
}

std::vector<TaskLoop*> TaskLoopThreadPool::GetAllLoops() {
  PADDLE_ENFORCE_EQ(start_, true, platform::errors::PreconditionNotMet(
                                      "thread pool must start first."));
  return loops_;
}

}  // namespace distributed
}  // namespace paddle
