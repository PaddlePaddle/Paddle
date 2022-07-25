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

#include <mpi-ext.h>
#include <iostream>
#include <limits>
#include <map>

#include "paddle/fluid/distributed/collective/Common.h"
#include "paddle/fluid/distributed/collective/MPITools.h"
#include "paddle/fluid/distributed/collective/ProcessGroupMPI.h"

constexpr int64_t kWaitBlockTImeout = 10;

namespace paddle {
namespace distributed {

void ProcessGroupMPI::MPITask::finishMPITaskError(std::exception_ptr eptr) {
  finish(eptr);
}

void ProcessGroupMPI::MPITask::finishMPITask() { finish(); }

ProcessGroupMPI::MPIAsyncTask::MPIAsyncTask(
    MPI_Request request, const std::vector<phi::DenseTensor>& inputs)
    : ProcessGroup::Task(-1, inputs, CommType::UNKNOWN), request_(request) {
  memset(&status_, 0, sizeof(status_));
}

ProcessGroupMPI::MPIAsyncTask::~MPIAsyncTask() {
  if (request_ != MPI_REQUEST_NULL) {
    std::cerr << " Task has not completed, try to destruct async mpi task, "
              << "exit the program." << std::endl;
    std::terminate();
  }
}

bool ProcessGroupMPI::MPIAsyncTask::isCompleted() {
  if (request_ == MPI_REQUEST_NULL) {
    return true;
  }

  std::unique_lock<std::mutex> globalLock(pg_global_mutex);
  int flag = 0;
  MPI_CHECK(MPI_Test(&request_, &flag, &status_));
  if (request_ != MPI_REQUEST_NULL) {
    return false;
  }

  if (status_.MPI_ERROR != MPI_SUCCESS) {
    appearException();
  }

  return true;
}

bool ProcessGroupMPI::MPIAsyncTask::Wait(std::chrono::milliseconds timeout) {
  if (request_ == MPI_REQUEST_NULL) {
    return true;
  }

  std::unique_lock<std::mutex> globalLock(pg_global_mutex);
  MPI_CHECK(MPI_Wait(&request_, &status_));

  if (status_.MPI_ERROR != MPI_SUCCESS) {
    appearException();
    std::rethrow_exception(exception_);
  }

  return true;
}

void ProcessGroupMPI::MPIAsyncTask::appearException() {
  std::array<char, MPI_MAX_ERROR_STRING> buf;
  int len = buf.size();
  MPI_CHECK(MPI_Error_string(status_.MPI_ERROR, buf.data(), &len));
  exception_ =
      std::make_exception_ptr(std::runtime_error(std::string(buf.data(), len)));
}

void ProcessGroupNCCL::MPIAsyncTask::SetOutputs(
    std::vector<phi::DenseTensor>& outputs) {  // NOLINT
  outputs_ = std::make_shared<std::vector<phi::DenseTensor>>(outputs);
}

// some global static var
int ProcessGroupMPI::mpi_thread_support = 0;
std::mutex ProcessGroupMPI::pg_global_mutex;
std::once_flag ProcessGroupMPI::onceFlag;

void ProcessGroupMPI::exitMPI() {
  std::unique_lock<std::mutex> globalLock(pg_global_mutex);
  MPI_CHECK(MPI_Finalize());
}

void ProcessGroupMPI::initOneTimeMPI() {
  std::call_once(onceFlag, []() {
    MPI_CHECK(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED,
                              &mpi_thread_support));

    PADDLE_ENFORCE_EQ(
        mpi_thread_support < MPI_THREAD_SERIALIZED, false,
        platform::errors::InvalidArgument("MPI supports the number of threads "
                                          "less than MPI_THREAD_SERIALIZED. "));

    std::atexit(ProcessGroupMPI::exitMPI);
  });
}

static std::shared_ptr<ProcessGroupMPI> ProcessGroupMPI::createProcessGroupMPI(
    std::vector<int> ranks, int gid) {
  // init once mpi
  initOneTimeMPI();

  MPI_Comm groupComm = MPI_COMM_WORLD;

  {
    std::lock_guard<std::mutex> globalLock(pg_global_mutex);

    // If no ranks are specified, assume we're creating the root group
    if (!ranks.empty()) {
      MPI_Group worldGroup;
      MPI_Group ranksGroup;
      MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &worldGroup));
      MPI_CHECK(
          MPI_Group_incl(worldGroup, ranks.size(), ranks.data(), &ranksGroup));

      constexpr int maxRetries = 3;
      bool create_success = false;
      MPI_Barrier(MPI_COMM_WORLD);
      for (auto i = 0; i < maxRetries; i++) {
        (void)i;
        if (MPI_Comm_create(MPI_COMM_WORLD, ranksGroup, &groupComm)) {
          create_success = true;
          break;
        }
      }
      MPI_CHECK(create_success);
      MPI_CHECK(MPI_Group_free(&worldGroup));
      MPI_CHECK(MPI_Group_free(&ranksGroup));
    }

    // Fetch rank and world size for this group (MPI_COMM_WORLD or new)
    int rank = -1, size = -1;
    if (groupComm != MPI_COMM_NULL) {
      MPI_CHECK(MPI_Comm_rank(groupComm, &rank));
      MPI_CHECK(MPI_Comm_size(groupComm, &size));

      PADDLE_ENFORCE_EQ(
          rank < 0 || size < 0, false,
          platform::errors::InvalidArgument("get world_size or rank failed!"));
    }
  }

  if (groupComm == MPI_COMM_NULL) {
    return std::shared_ptr<ProcessGroupMPI>();
  }

  return std::shared_ptr<ProcessGroupMPI>(rank, size, groupComm, gid);
}

ProcessGroupMPI::ProcessGroupMPI(int rank, int size, MPI_Comm pg_comm, int gid)
    : ProcessGroup(rank, size, gid), stop_(false), pg_comm(pg_comm) {
  PADDLE_ENFORCE_EQ(
      pg_comm == MPI_COMM_NULL, false,
      platform::errors::InvalidArgument("Error! mpi comm is MPI_COMM_NULL!"));

  worker_thread = std::thread(&ProcessGroupMPI::workLoop, this);
}

ProcessGroupMPI::~ProcessGroupMPI() {
  std::unique_lock<std::mutex> lock(pg_mutex);
  queue_consume.wait(lock, [&] { return queue_.empty(); });
  stop_ = true;
  lock.unlock();
  queue_produce.notify_all();

  worker_thread.join();
}

void ProcessGroupMPI::workLoop() {
  std::unique_lock<std::mutex> lock(pg_mutex);

  while (!stop_) {
    if (queue_.empty()) {
      queue_produce.wait(lock);
      continue;
    }

    auto taskTuple = std::move(queue_.front());

    queue_.pop_front();

    auto& taskEntry = std::get<0>(taskTuple);
    auto& task = std::get<1>(taskTuple);

    lock.unlock();
    queue_consume.notify_one();

    try {
      taskEntry->run(taskEntry);
      task->finishWorkMPI();
    } catch (...) {
      task->finishWorkMPIError(std::current_exception());
    }

    lock.lock();
  }
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupMPI::enqueue(
    std::unique_ptr<TaskEntry> entry,
    const std::vector<phi::DenseTensor>& inputs) {
  auto task = std::shared_ptr<MPITask>(entry->dst, inputs);
  std::unique_lock<std::mutex> lock(pg_mutex);
  queue_.push_back(std::make_tuple(std::move(entry), task));
  lock.unlock();
  queue_produce.notify_one();
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupMPI::Broadcast(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const BroadcastOptions& opts = BroadcastOptions()) {
  CheckValidInputs(in_tensors);
  const auto places = GetPlaceList(in_tensors);

  std::function<void(std::unique_ptr<TaskEntry>&)> runFunc =
      [opts, this](std::unique_ptr<TaskEntry>& entry) {
        auto data = (entry->src)[0];
        DeviceGuard guard(data.place());
        std::unique_lock<std::mutex> globalLock(pg_global_mutex);
        MPI_CHECK(MPI_Bcast(data.data_ptr(), data.numel(),
                            mpiDatatype.at(data.scalar_type()), opts.rootRank,
                            pg_comm));
      };
  auto entry = std::make_unique<TaskEntry>(&in_tensors, &out_tensors,
                                           std::move(runFunc));
  return enqueue(std::move(entry), in_tensors);
}

}  //  namespace distributed
}  //  namespace paddle
