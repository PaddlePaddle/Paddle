/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#if defined(PADDLE_WITH_PSCORE)
#include "paddle/fluid/distributed/ps/wrapper/fleet.h"
#endif

#include "paddle/fluid/framework/threadpool.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/trainer.h"

namespace paddle::framework {

void DistMultiTrainer::Initialize(const TrainerDesc &trainer_desc,
                                  Dataset *dataset) {
  thread_num_ = trainer_desc.thread_num();
  SetDataset(dataset);

  ParseDumpConfig(trainer_desc);
  mpi_rank_ = trainer_desc.mpi_rank();
  mpi_size_ = trainer_desc.mpi_size();
  dump_file_num_ = trainer_desc.dump_file_num();
  user_define_dump_filename_ = trainer_desc.user_define_dump_filename();
  const std::vector<paddle::framework::DataFeed *> readers =
      dataset->GetReaders();
  RegisterHeterCallback();
  thread_num_ = static_cast<int>(readers.size());
  workers_.resize(thread_num_);
  for (int i = 0; i < trainer_desc.downpour_param().stat_var_names_size();
       i++) {
    need_merge_var_names_.push_back(
        trainer_desc.downpour_param().stat_var_names(i));
  }

  for (int i = 0; i < thread_num_; ++i) {
    workers_[i] = DeviceWorkerFactory::CreateDeviceWorker(
        trainer_desc.device_worker_name());
    workers_[i]->SetDeviceIndex(i);
    workers_[i]->SetDataFeed(readers[i]);
    workers_[i]->SetNeedDumpField(need_dump_field_);
    workers_[i]->SetNeedDumpParam(need_dump_param_);
    workers_[i]->SetDumpFieldVector(dump_fields_);
    workers_[i]->SetDumpParamVector(dump_param_);
    workers_[i]->InitRandomDumpConfig(trainer_desc);
    workers_[i]->Initialize(trainer_desc);
    workers_[i]->SetWorkerNum(thread_num_);
  }

  VLOG(3) << "going to initialize pull dense worker";
  pull_dense_worker_ = PullDenseWorker::GetInstance();
  pull_dense_worker_->Initialize(trainer_desc);
  VLOG(3) << "initialize pull dense worker";
  SetDebug(trainer_desc.debug());
}

void DistMultiTrainer::RegisterHeterCallback() {
#ifdef PADDLE_WITH_PSCORE
  auto fleet_ptr = paddle::distributed::FleetWrapper::GetInstance();
#else
  auto fleet_ptr = FleetWrapper::GetInstance();
#endif
  fleet_ptr->RegisterHeterCallback(
      [this](int worker, int taskid) { workers_[worker]->Schedule(taskid); });
}

void DistMultiTrainer::InitDumpEnv() {
  queue_ = paddle::framework::MakeChannel<std::string>();
  for (int i = 0; i < thread_num_; ++i) {
    workers_[i]->SetChannelWriter(queue_.get());
  }
  dump_thread_num_ = 1;
  if (dump_file_num_ > mpi_size_) {
    dump_thread_num_ = dump_file_num_ / mpi_size_;
    if (dump_file_num_ % mpi_size_ > mpi_rank_) {
      dump_thread_num_ += 1;
    }
  }
  for (int i = 0; i < dump_thread_num_; i++) {
    dump_thread_.emplace_back([this, i] { DumpWork(i); });
  }
}

inline std::vector<std::shared_ptr<phi::ThreadPool>> &GetThreadPool(
    int thread_num) {
  static std::vector<std::shared_ptr<phi::ThreadPool>> thread_pools;
  if (!thread_pools.empty()) {
    return thread_pools;
  }
  thread_pools.resize(thread_num);
  for (int i = 0; i < thread_num; ++i) {
    thread_pools[i].reset(new phi::ThreadPool(1));
  }
  return thread_pools;
}

void DistMultiTrainer::InitTrainerEnv(const ProgramDesc &main_program,
                                      const phi::Place &place) {
  auto pool = GetThreadPool(thread_num_);
  std::vector<std::future<void>> wait_futures;
  PADDLE_ENFORCE_EQ(static_cast<int>(pool.size()),
                    thread_num_,
                    common::errors::InvalidArgument(
                        "static_cast<int>(pool.size()) is invalid, "
                        "expected %d but received %d.",
                        thread_num_,
                        static_cast<int>(pool.size())));
  for (int i = 0; i < thread_num_; ++i) {
    wait_futures.emplace_back(pool[i]->Run([this, i, &main_program, &place]() {
      workers_[i]->SetPlace(place);
      workers_[i]->SetReaderPlace(place);
      workers_[i]->SetRootScope(root_scope_);
      workers_[i]->CreateDeviceResource(main_program);  // Program
      workers_[i]->BindingDataFeedMemory();
#if defined(PADDLE_WITH_PSLIB) || defined(PADDLE_WITH_PSCORE)
      workers_[i]->CacheProgram(main_program);
#endif
    }));
  }
  for (auto &th : wait_futures) {
    th.get();
  }
  // Scope* -> thread id, it will be used in push_dense op
  for (int i = 0; i < thread_num_; ++i) {
    Scope *thread_scope = workers_[i]->GetThreadScope();
    pull_dense_worker_->SetThreadIdByScope(thread_scope, i);
  }
}

void DistMultiTrainer::InitOtherEnv(const ProgramDesc &main_program) {
  if (need_dump_field_ || need_dump_param_) {
    InitDumpEnv();
  }
  pull_dense_worker_->SetRootScope(root_scope_);
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_CUDA)
  pull_dense_worker_->CreatePinVar();
#endif
  pull_dense_worker_->Start();
#if defined(PADDLE_WITH_PSLIB) || defined(PADDLE_WITH_PSCORE)
  for (int i = 0; i < thread_num_; ++i) {
    workers_[i]->GetXpuOpIndex();
  }
#endif
  VLOG(3) << "init other env done.";
}

void DistMultiTrainer::Run() {
  auto pool = GetThreadPool(thread_num_);
  std::vector<std::future<void>> wait_futures;
  PADDLE_ENFORCE_EQ(static_cast<int>(pool.size()),
                    thread_num_,
                    common::errors::InvalidArgument(
                        "static_cast<int>(pool.size()) is invalid, "
                        "expected %d but received %d.",
                        thread_num_,
                        static_cast<int>(pool.size())));
  for (int i = 0; i < thread_num_; ++i) {
    if (!debug_) {  // NOLINT
      wait_futures.emplace_back(
          pool[i]->Run([this, i]() { workers_[i]->TrainFiles(); }));
    } else {
      wait_futures.emplace_back(
          pool[i]->Run([this, i]() { workers_[i]->TrainFilesWithProfiler(); }));
    }
  }
  for (auto &th : wait_futures) {
    th.get();
  }
}

Scope *DistMultiTrainer::GetWorkerScope(int thread_id) {
  return workers_[thread_id]->GetThreadScope();
}

void DistMultiTrainer::Finalize() {
  for (size_t i = 0; i < need_merge_var_names_.size(); i++) {
    Variable *root_var = root_scope_->FindVar(need_merge_var_names_[i]);
    if (root_var == nullptr) {
      continue;
    }
    phi::DenseTensor *root_tensor = root_var->GetMutable<phi::DenseTensor>();
    for (int j = 1; j < thread_num_; j++) {
      Scope *cur_thread_scope = workers_[j]->GetThreadScope();
      Variable *thread_var =
          cur_thread_scope->FindVar(need_merge_var_names_[i]);
      phi::DenseTensor *thread_tensor =
          thread_var->GetMutable<phi::DenseTensor>();
      if (root_tensor->numel() != thread_tensor->numel()) {
        continue;
      }
#define MergeCallback(cpp_type, proto_type)                                    \
  do {                                                                         \
    if (framework::TransToProtoVarType(root_tensor->dtype()) == proto_type) {  \
      if (framework::TransToProtoVarType(thread_tensor->dtype()) !=            \
          proto_type) {                                                        \
        VLOG(0) << "Error: thread id=" << j << ", need_merge_var_names_[" << i \
                << "] " << need_merge_var_names_[i]                            \
                << ", root tensor type=" << root_tensor->dtype()               \
                << ", thread tensor type=" << thread_tensor->dtype();          \
        exit(-1);                                                              \
      }                                                                        \
      MergeToRootScope<cpp_type>(root_tensor, thread_tensor);                  \
    }                                                                          \
  } while (0)
      _ForEachDataType_(MergeCallback);
    }
  }

  if (need_dump_field_ || need_dump_param_) {
    FinalizeDumpEnv();
  }
  pull_dense_worker_->Stop();
  root_scope_->DropKids();

// flush local client push queue
#ifdef PADDLE_WITH_PSCORE
  auto fleet_ptr_ = paddle::distributed::FleetWrapper::GetInstance();
#else
  auto fleet_ptr_ = FleetWrapper::GetInstance();
#endif
  fleet_ptr_->ClientFlush();
}

template <typename T>
void DistMultiTrainer::MergeToRootScope(phi::DenseTensor *root_tensor,
                                        phi::DenseTensor *tensor) {
  T *root_data = root_tensor->data<T>();
  T *data = tensor->data<T>();
  for (int i = 0; i < tensor->numel(); i++) {
    root_data[i] += data[i];
  }
}
}  // namespace paddle::framework
