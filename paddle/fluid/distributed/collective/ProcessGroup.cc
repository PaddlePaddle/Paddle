// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/collective/ProcessGroup.h"

namespace paddle {
namespace distributed {

ProcessGroup::Task::Task(int rank,
                         const std::vector<phi::DenseTensor>& inputTensors,
                         CommType comm_type)
    : rank_(rank), comm_type_(comm_type) {}

ProcessGroup::Task::~Task() = default;

bool ProcessGroup::Task::IsCompleted() {
  std::lock_guard<std::mutex> lock(mutex_);
  return is_completed_;
}

bool ProcessGroup::Task::Wait(std::chrono::milliseconds timeout) {
  return false;
}

void ProcessGroup::Task::Synchronize() {}

ProcessGroup::ProcessGroup(int rank, int size, const platform::Place& place,
                           int gid)
    : rank_(rank), size_(size), place_(place), gid_(gid) {
  if (gid != IGNORE_ID) {
    auto map = ProcessGroupMapFromGid::getInstance();
    map->insert(gid_, this);
  }
}

}  //  namespace distributed
}  //  namespace paddle
