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

#include <functional>
#include <memory>

namespace paddle {
namespace framework {

class WorkQueue {
 public:
  WorkQueue() = default;

  WorkQueue(const WorkQueue&) = delete;
  
  WorkQueue& operator=(const WorkQueue&) = delete;

  virtual ~WorkQueue() = default;

  virtual void AddTask(std::function<void()> fn) = 0;

  virtual void WaitQueueEmpty() = 0;

  virtual size_t NumThreads() = 0; 
};

std::unique_ptr<WorkQueue> CreateSingleThreadedWorkQueue();

std::unique_ptr<WorkQueue> CreateMultiThreadedWorkQueue(int num_threads);

}  // namespace framework
}  // namespace paddle
