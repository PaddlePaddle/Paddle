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
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace distributed {

std::string commTypeToString(CommType commType) {
  switch (commType) {
    case CommType::BROADCAST:
      return "BROADCAST";
    case CommType::ALLREDUCE:
      return "ALLREDUCE";
    case CommType::ALLREDUCE_SPARSE:
      return "ALLREDUCE_SPARSE";
    case CommType::REDUCE:
      return "REDUCE";
    case CommType::ALLGATHER:
      return "ALLGATHER";
    case CommType::GATHER:
      return "GATHER";
    case CommType::SCATTER:
      return "SCATTER";
    case CommType::REDUCE_SCATTER:
      return "REDUCE_SCATTER";
    case CommType::ALLTOALL:
      return "ALLTOALL";
    case CommType::SEND:
      return "SEND";
    case CommType::RECV:
      return "RECV";
    case CommType::BARRIER:
      return "BARRIER";
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unknown communication type (CommType)."));
  }
  return "UNKNOWN";
}

ProcessGroup::Task::Task(int rank, const CommType comm_type)
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

ProcessGroup::ProcessGroup(int rank, int size) : rank_(rank), size_(size) {}

}  //  namespace distributed
}  //  namespace paddle
