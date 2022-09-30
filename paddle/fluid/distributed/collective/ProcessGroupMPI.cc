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

#include "paddle/fluid/distributed/collective/ProcessGroupMPI.h"
#include <chrono>
#include "paddle/fluid/distributed/collective/Common.h"

constexpr int64_t kWaitBlockTImeout = 10;
namespace paddle {
namespace distributed {

std::map<phi::DataType, MPI_Datatype> mpiDatatype = {
    {phi::DataType::INT8, MPI_CHAR},
    {phi::DataType::UINT8, MPI_UNSIGNED_CHAR},
    {phi::DataType::FLOAT32, MPI_FLOAT},
    {phi::DataType::FLOAT64, MPI_DOUBLE},
    {phi::DataType::INT32, MPI_INT},
    {phi::DataType::INT64, MPI_LONG}};

void ProcessGroupMPI::MPITask::FinishMPITaskError(std::exception_ptr eptr) {
  Finish(eptr);
}

void ProcessGroupMPI::MPITask::FinishMPITask() { Finish(); }

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

bool ProcessGroupMPI::MPIAsyncTask::IsCompleted() {
  if (request_ == MPI_REQUEST_NULL) {
    return true;
  }

  std::unique_lock<std::mutex> lock(pg_global_mutex);
  int flag = 0;
  MPI_CHECK(MPI_Test(&request_, &flag, &status_));
  if (request_ != MPI_REQUEST_NULL) {
    return false;
  }

  if (status_.MPI_ERROR != MPI_SUCCESS) {
    AppearException();
  }

  return true;
}

bool ProcessGroupMPI::MPIAsyncTask::Wait(std::chrono::milliseconds timeout) {
  if (request_ == MPI_REQUEST_NULL) {
    return true;
  }

  std::unique_lock<std::mutex> lock(pg_global_mutex);
  MPI_CHECK(MPI_Wait(&request_, &status_));

  if (status_.MPI_ERROR != MPI_SUCCESS) {
    AppearException();
    std::rethrow_exception(exception_);
    return false;
  }

  return true;
}

void ProcessGroupMPI::MPIAsyncTask::AppearException() {
  std::array<char, MPI_MAX_ERROR_STRING> buf;
  int len = buf.size();
  MPI_CHECK(MPI_Error_string(status_.MPI_ERROR, buf.data(), &len));
  exception_ =
      std::make_exception_ptr(std::runtime_error(std::string(buf.data(), len)));
}

void ProcessGroupMPI::MPIAsyncTask::SetOutputs(
    std::vector<phi::DenseTensor>& outputs) {
  outputs_ = std::make_shared<std::vector<phi::DenseTensor>>(outputs);
}

int ProcessGroupMPI::mpi_thread_support = 0;
std::mutex ProcessGroupMPI::pg_global_mutex;
std::once_flag ProcessGroupMPI::onceFlag;

void ProcessGroupMPI::ExitMPI() {
  std::unique_lock<std::mutex> lock(pg_global_mutex);
  MPI_CHECK(MPI_Finalize());
}

void ProcessGroupMPI::InitOneTimeMPI() {
  std::call_once(onceFlag, []() {
    MPI_CHECK(MPI_Init_thread(
        nullptr, nullptr, MPI_THREAD_SERIALIZED, &mpi_thread_support));
    PADDLE_ENFORCE_EQ(
        mpi_thread_support < MPI_THREAD_SERIALIZED,
        false,
        platform::errors::InvalidArgument("MPI supports the number of threads "
                                          "less than MPI_THREAD_SERIALIZED. "));

    std::atexit(ProcessGroupMPI::ExitMPI);
  });
}

std::shared_ptr<ProcessGroupMPI> ProcessGroupMPI::CreateProcessGroupMPI(
    const std::vector<int>& ranks, int gid) {
  InitOneTimeMPI();

  MPI_Comm groupComm = MPI_COMM_WORLD;
  int rank = -1;
  int size = -1;

  {
    std::lock_guard<std::mutex> lock(pg_global_mutex);

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
        if (MPI_Comm_create(MPI_COMM_WORLD, ranksGroup, &groupComm)) {
          create_success = true;
          break;
        }
      }
      MPI_CHECK(create_success);
      MPI_CHECK(MPI_Group_free(&worldGroup));
      MPI_CHECK(MPI_Group_free(&ranksGroup));
    }

    if (groupComm != MPI_COMM_NULL) {
      MPI_CHECK(MPI_Comm_rank(groupComm, &rank));
      MPI_CHECK(MPI_Comm_size(groupComm, &size));

      PADDLE_ENFORCE_EQ(
          rank < 0 || size < 0,
          false,
          platform::errors::InvalidArgument("get world_size or rank failed!"));
    }
  }

  if (groupComm == MPI_COMM_NULL) {
    return std::shared_ptr<ProcessGroupMPI>();
  }

  VLOG(3) << "MPI Group Create Success! rank = " << rank << " size = " << size
          << " group_id = " << gid;

  return std::make_shared<ProcessGroupMPI>(rank, size, groupComm, gid);
}

ProcessGroupMPI::ProcessGroupMPI(int rank, int size, MPI_Comm pg_comm, int gid)
    : ProcessGroup(rank, size, gid), stop_(false), pg_comm(pg_comm) {
  PADDLE_ENFORCE_EQ(
      pg_comm == MPI_COMM_NULL,
      false,
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
      taskEntry->run_(taskEntry);
      task->FinishMPITask();
    } catch (...) {
      task->FinishMPITaskError(std::current_exception());
    }

    lock.lock();
  }
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupMPI::Enqueue(
    std::unique_ptr<TaskEntry> entry,
    const std::vector<phi::DenseTensor>& inputs) {
  auto task = std::make_shared<MPITask>(entry->dst_, inputs);
  std::unique_lock<std::mutex> lock(pg_mutex);
  queue_.push_back(std::make_tuple(std::move(entry), task));
  lock.unlock();
  queue_produce.notify_one();
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupMPI::Broadcast(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const BroadcastOptions& opts) {
  mpi::CheckValidInputs(in_tensors);
  const auto places = GetPlaceList(in_tensors);

  std::function<void(std::unique_ptr<TaskEntry>&)> runFunc =
      [opts, this](std::unique_ptr<TaskEntry>& entry) {
        auto data = (entry->src_)[0];
        std::unique_lock<std::mutex> lock(pg_global_mutex);
        const auto root = opts.source_rank + opts.source_root;
        MPI_CHECK(MPI_Bcast(data.data(),
                            data.numel(),
                            mpiDatatype.at(data.dtype()),
                            root,
                            pg_comm));
      };
  auto entry = std::make_unique<TaskEntry>(
      &in_tensors, &out_tensors, std::move(runFunc));
  return Enqueue(std::move(entry), in_tensors);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupMPI::AllReduce(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const AllreduceOptions& opts) {
  mpi::CheckValidInputs(in_tensors);

  std::function<void(std::unique_ptr<TaskEntry>&)> runFunc =
      [opts, this](std::unique_ptr<TaskEntry>& entry) {
        auto data = (entry->src_)[0];
        std::unique_lock<std::mutex> lock(pg_global_mutex);
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,
                                data.data(),
                                data.numel(),
                                mpiDatatype.at(data.dtype()),
                                mpi::ToMPIType(opts.reduce_op),
                                pg_comm));
      };
  auto entry = std::make_unique<TaskEntry>(
      &in_tensors, &out_tensors, std::move(runFunc));
  return Enqueue(std::move(entry), in_tensors);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupMPI::Barrier(
    const BarrierOptions& opts) {
  std::function<void(std::unique_ptr<TaskEntry>&)> runFunc =
      [this](std::unique_ptr<TaskEntry>& entry) {
        std::unique_lock<std::mutex> lock(pg_global_mutex);
        MPI_CHECK(MPI_Barrier(pg_comm));
      };
  auto entry =
      std::make_unique<TaskEntry>(nullptr, nullptr, std::move(runFunc));
  return Enqueue(std::move(entry), std::vector<phi::DenseTensor>{});
}

// NOTE: MPI_send tag set gid_
std::shared_ptr<ProcessGroup::Task> ProcessGroupMPI::Send(
    std::vector<phi::DenseTensor>& tensors, int dst_rank) {
  mpi::CheckValidInputs(tensors);

  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    std::unique_lock<std::mutex> lock(pg_global_mutex);
    MPI_CHECK(MPI_Isend(tensor.data(),
                        tensor.numel(),
                        mpiDatatype.at(tensor.dtype()),
                        dst_rank,
                        this->gid_,
                        pg_comm,
                        &request));
  }

  return std::make_shared<ProcessGroupMPI::MPIAsyncTask>(request, tensors);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupMPI::Recv(
    std::vector<phi::DenseTensor>& tensors, int src_rank) {
  mpi::CheckValidInputs(tensors);

  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    std::unique_lock<std::mutex> lock(pg_global_mutex);
    MPI_CHECK(MPI_Irecv(tensor.data(),
                        tensor.numel(),
                        mpiDatatype.at(tensor.dtype()),
                        src_rank,
                        this->gid_,
                        pg_comm,
                        &request));
  }

  return std::make_shared<ProcessGroupMPI::MPIAsyncTask>(request, tensors);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupMPI::AllGather(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors) {
  mpi::CheckValidInputs(in_tensors);

  PADDLE_ENFORCE_EQ(out_tensors.size() == 1,
                    true,
                    platform::errors::InvalidArgument(
                        "MPI only support a single tensor op."));

  std::function<void(std::unique_ptr<TaskEntry>&)> runFunc =
      [this](std::unique_ptr<TaskEntry>& entry) {
        auto data = (entry->src_)[0];
        std::vector<phi::DenseTensor> dst = entry->dst_;

        std::unique_lock<std::mutex> lock(pg_global_mutex);
        MPI_CHECK(MPI_Allgather(data.data(),
                                data.numel(),
                                mpiDatatype.at(data.dtype()),
                                dst[0].data(),
                                data.numel(),
                                mpiDatatype.at(data.dtype()),
                                pg_comm));
      };

  auto entry = std::make_unique<TaskEntry>(
      &in_tensors, &out_tensors, std::move(runFunc));

  return Enqueue(std::move(entry), in_tensors);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupMPI::AllToAll(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors) {
  mpi::CheckValidInputs(in_tensors);
  mpi::CheckValidInputs(out_tensors);

  PADDLE_ENFORCE_EQ(in_tensors[0].numel() == out_tensors[0].numel() &&
                        in_tensors[0].dtype() == out_tensors[0].dtype(),
                    true,
                    platform::errors::InvalidArgument(
                        "MPI AlltoAll: input and output are not equal in "
                        "size or data type."));

  std::function<void(std::unique_ptr<TaskEntry>&)> runFunc =
      [this](std::unique_ptr<TaskEntry>& entry) {
        auto srcdata = (entry->src_)[0];
        auto dstdata = (entry->dst_)[0];
        std::unique_lock<std::mutex> lock(pg_global_mutex);
        MPI_CHECK(MPI_Alltoall(srcdata.data(),
                               srcdata.numel() / size_,
                               mpiDatatype.at(srcdata.dtype()),
                               dstdata.data(),
                               dstdata.numel() / size_,
                               mpiDatatype.at(dstdata.dtype()),
                               pg_comm));
      };
  auto entry = std::make_unique<TaskEntry>(
      &in_tensors, &out_tensors, std::move(runFunc));

  return Enqueue(std::move(entry), in_tensors);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupMPI::Reduce(
    std::vector<phi::DenseTensor>& tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const ReduceOptions& opts) {
  mpi::CheckValidInputs(tensors);

  std::function<void(std::unique_ptr<TaskEntry>&)> runFunc =
      [opts, this](std::unique_ptr<TaskEntry>& entry) {
        auto data = (entry->src_)[0];
        auto dataPtr = (entry->src_)[0].data();
        void* sendbuf = (rank_ == opts.root_rank) ? MPI_IN_PLACE : dataPtr;
        void* recvbuf = (rank_ == opts.root_rank) ? dataPtr : nullptr;

        std::unique_lock<std::mutex> lock(pg_global_mutex);
        MPI_CHECK(MPI_Reduce(sendbuf,
                             recvbuf,
                             data.numel(),
                             mpiDatatype.at(data.dtype()),
                             mpi::ToMPIType(opts.reduce_op),
                             opts.root_rank,
                             pg_comm));
      };
  auto entry =
      std::make_unique<TaskEntry>(&tensors, &tensors, std::move(runFunc));
  return Enqueue(std::move(entry), tensors);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupMPI::Scatter(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const ScatterOptions& opts) {
  mpi::CheckValidInputs(in_tensors);

  std::function<void(std::unique_ptr<TaskEntry>&)> runFunc =
      [opts, this](std::unique_ptr<TaskEntry>& entry) {
        auto data = (entry->dst_)[0];
        void* sendbuf = nullptr;

        if (rank_ == opts.root_rank) {
          std::vector<phi::DenseTensor>& inputData = entry->src_;
          sendbuf = inputData[0].data();
        }

        std::unique_lock<std::mutex> lock(pg_global_mutex);
        MPI_CHECK(MPI_Scatter(sendbuf,
                              data.numel(),
                              mpiDatatype.at(data.dtype()),
                              data.data(),
                              data.numel(),
                              mpiDatatype.at(data.dtype()),
                              opts.root_rank,
                              pg_comm));
      };

  if (rank_ == opts.root_rank) {
    auto entry = std::make_unique<TaskEntry>(
        &in_tensors, &out_tensors, std::move(runFunc));
    return Enqueue(std::move(entry), in_tensors);
  } else {
    auto entry =
        std::make_unique<TaskEntry>(nullptr, &out_tensors, std::move(runFunc));
    return Enqueue(std::move(entry), in_tensors);
  }
}

}  //  namespace distributed
}  //  namespace paddle
