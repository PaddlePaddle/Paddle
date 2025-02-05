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

#include <string>

#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/platform/densetensor_printer.h"
#if defined PADDLE_WITH_PSCORE
#include "paddle/fluid/distributed/ps/service/communicator/communicator.h"
#endif
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/program_utils.h"

PHI_DEFINE_EXPORTED_bool(enable_dump_main_program,
                         false,
                         "enable dump main program, default false");

namespace paddle::framework {

extern Barrier g_barrier;

void MultiTrainer::Initialize(const TrainerDesc& trainer_desc,
                              Dataset* dataset) {
  thread_num_ = trainer_desc.thread_num();
  SetDataset(dataset);

  ParseDumpConfig(trainer_desc);
  mpi_rank_ = trainer_desc.mpi_rank();
  mpi_size_ = trainer_desc.mpi_size();
  dump_file_num_ = trainer_desc.dump_file_num();
  for (int i = 0; i < trainer_desc.downpour_param().stat_var_names_size();
       i++) {
    need_merge_var_names_.push_back(
        trainer_desc.downpour_param().stat_var_names(i));
  }
  use_ps_gpu_ = trainer_desc.use_ps_gpu();
  use_gpu_graph_ = trainer_desc.use_gpu_graph();
  VLOG(3) << "Initialize use_ps_gpu_:" << use_ps_gpu_
          << "; use_gpu_graph_:" << use_gpu_graph_;
#ifdef PADDLE_WITH_HETERPS
  for (int i = 0; i < thread_num_; ++i) {
    int num = trainer_desc.worker_places(i);
    phi::GPUPlace place = phi::GPUPlace(num);
    places_.push_back(place);
  }
#endif
  user_define_dump_filename_ = trainer_desc.user_define_dump_filename();
  // get filelist from trainer_desc here
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();
  VLOG(3) << "readers num: " << readers.size();
  // change thread num to readers num
  thread_num_ = static_cast<int>(readers.size());
  VLOG(3) << "worker thread num: " << thread_num_;
  workers_.resize(thread_num_);

#if defined PADDLE_WITH_PSCORE
  if (trainer_desc.thread_barrier()) {
    paddle::distributed::Communicator::GetInstance()->BarrierTriggerReset(
        thread_num_);
  }
#endif
  g_barrier.reset(thread_num_);
  for (int i = 0; i < thread_num_; ++i) {
    workers_[i] = DeviceWorkerFactory::CreateDeviceWorker(
        trainer_desc.device_worker_name());
    workers_[i]->SetNeedDumpField(need_dump_field_);
    workers_[i]->SetNeedDumpParam(need_dump_param_);
    workers_[i]->SetDumpFieldVector(dump_fields_);
    workers_[i]->SetDumpParamVector(dump_param_);
    workers_[i]->InitRandomDumpConfig(trainer_desc);
    workers_[i]->Initialize(trainer_desc);
    workers_[i]->SetDeviceIndex(i);
    workers_[i]->SetDataFeed(readers[i]);
    workers_[i]->SetThreadNum(thread_num_);
  }

  // set debug here
  SetDebug(trainer_desc.debug());
}

std::string MultiTrainer::GetDumpPath(int tid) {
  if (!user_define_dump_filename_.empty()) {
    return string::format_string("%s/part-%s-%05d",
                                 dump_fields_path_.c_str(),
                                 user_define_dump_filename_.c_str(),
                                 tid);
  }
  return string::format_string(
      "%s/part-%03d-%05d", dump_fields_path_.c_str(), mpi_rank_, tid);
}

void MultiTrainer::InitDumpEnv() {
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
inline std::vector<std::shared_ptr<phi::ThreadPool>>& GetThreadPool(
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
// call only after all resources are set in current trainer
void MultiTrainer::InitTrainerEnv(const ProgramDesc& main_program,
                                  const phi::Place& place) {
  // multi thread load
  auto pool = GetThreadPool(thread_num_);
  std::vector<std::future<void>> wait_futures;
  PADDLE_ENFORCE_EQ(static_cast<int>(pool.size()),
                    thread_num_,
                    common::errors::InvalidArgument(
                        "static_cast<int>(pool.size()) is invalid, "
                        "expected %d but received %d",
                        thread_num_,
                        static_cast<int>(pool.size())));
  for (int i = 0; i < thread_num_; ++i) {
    wait_futures.emplace_back(pool[i]->Run([this, i, &main_program, &place]() {
#ifdef PADDLE_WITH_HETERPS
      workers_[i]->SetPlace(places_[i]);
      workers_[i]->SetReaderPlace(places_[i]);
      workers_[i]->SetDeviceContext(
          phi::DeviceContextPool::Instance().Get(places_[i]));
#else
      workers_[i]->SetPlace(place);
      workers_[i]->SetReaderPlace(place);
#endif
      workers_[i]->SetRootScope(root_scope_);
      workers_[i]->CreateDeviceResource(main_program);  // Program
      workers_[i]->BindingDataFeedMemory();
      workers_[i]->CacheProgram(main_program);
    }));
  }
  for (auto& th : wait_futures) {
    th.get();
  }

#ifdef PADDLE_WITH_HETERPS
  // only gpups mode
  if (!use_gpu_graph_ && use_ps_gpu_) {
    for (int num = 0; num < thread_num_; ++num) {
      auto place = places_[num];
      Scope* scope = workers_[num]->GetThreadScope();
      auto& block = main_program.Block(0);
      for (auto& var : block.AllVars()) {
        if (var->Persistable()) {
          auto name = var->Name();
          Variable* root_var = root_scope_->FindVar(name);
          if (!root_var) {
            continue;
          }
          if (root_var->IsType<phi::SelectedRows>()) {
            continue;
          }
          phi::DenseTensor* root_tensor =
              root_var->GetMutable<phi::DenseTensor>();
          auto* ptr = scope->Var(name);
          InitializeVariable(ptr, proto::VarType::DENSE_TENSOR);
          phi::DenseTensor* thread_tensor = ptr->GetMutable<phi::DenseTensor>();
          TensorCopy(*root_tensor, place, thread_tensor);
        }
      }
    }
  }
#endif
  if (!use_gpu_graph_) {  // cpups mode
    for (auto& var : main_program.Block(0).AllVars()) {
      if (var->Persistable()) {
        auto it = std::find(need_merge_var_names_.begin(),
                            need_merge_var_names_.end(),
                            var->Name());
        if (it == need_merge_var_names_.end() &&
            var->GetType() != proto::VarType::SELECTED_ROWS) {
          VLOG(2) << "train param: " << var->Name();
          trainable_param_.push_back(var->Name());
        }
      }
    }
  }
}

void MultiTrainer::InitOtherEnv(const ProgramDesc& main_program) {
  if (need_dump_field_ || need_dump_param_) {
    InitDumpEnv();
  }

#ifdef PADDLE_WITH_PSCORE
  // pull dense param first
  auto communicator = paddle::distributed::Communicator::GetInstance();
  // for unittest which call train_from_dataset but does not call
  // fleet.init_worker() first
  if (communicator == nullptr) {
    VLOG(1) << "MultiTrainer::InitOtherEnv Communicator is null!";
  } else {
    auto& recv_ctx = communicator->GetRecvCtxMap();
    communicator->PullDense(recv_ctx);
    VLOG(3) << "init other env done.";
  }
#endif
}

Scope* MultiTrainer::GetWorkerScope(int thread_id) {
  return workers_[thread_id]->GetThreadScope();
}
void MultiTrainer::Run() {
  VLOG(3) << "Going to run";
  auto pool = GetThreadPool(thread_num_);
  std::vector<std::future<void>> wait_futures;
  PADDLE_ENFORCE_EQ(static_cast<int>(pool.size()),
                    thread_num_,
                    common::errors::InvalidArgument(
                        "static_cast<int>(pool.size()) is invalid, "
                        "expected %d but received %d",
                        thread_num_,
                        static_cast<int>(pool.size())));
  for (int i = 0; i < thread_num_; ++i) {
    if (!debug_) {
      wait_futures.emplace_back(
          pool[i]->Run([this, i]() { workers_[i]->TrainFiles(); }));
    } else {
      wait_futures.emplace_back(
          pool[i]->Run([this, i]() { workers_[i]->TrainFilesWithProfiler(); }));
    }
  }
  for (auto& th : wait_futures) {
    th.get();
  }
  // merge worker vars
  MergeWorkerVars();
}

#ifdef PADDLE_WITH_HETERPS
void MultiTrainer::MergeDenseParam() {
#ifdef PADDLE_WITH_PSCORE
  auto communicator = paddle::distributed::Communicator::GetInstance();
  auto thread_scope = workers_[0]->GetThreadScope();
  if (communicator == nullptr) {
    for (auto& name : trainable_param_) {
      VLOG(2) << "merge var " << name << " to root scope";
      Variable* root_var = root_scope_->FindVar(name);
      phi::DenseTensor* root_tensor = root_var->GetMutable<phi::DenseTensor>();
      Variable* var = thread_scope->FindVar(name);
      phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
      TensorCopySync((*tensor), root_tensor->place(), root_tensor);
    }
  } else {
    auto& recv_ctx = communicator->GetRecvCtxMap();
    for (auto& iter : recv_ctx) {
      auto& varnames = iter.second;
      for (auto& name : varnames) {
        VLOG(2) << "merge var " << name << " to root scope";
        Variable* root_var = root_scope_->FindVar(name);
        phi::DenseTensor* root_tensor =
            root_var->GetMutable<phi::DenseTensor>();
        Variable* var = thread_scope->FindVar(name);
        phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
        TensorCopySync((*tensor), root_tensor->place(), root_tensor);
      }
    }
  }
#endif
}
#endif

template <typename T>
void MultiTrainer::MergeToRootScope(phi::DenseTensor* root_tensor,
                                    phi::DenseTensor* tensor) {
  phi::DenseTensor tmp_root;
  TensorCopy(*root_tensor, phi::CPUPlace(), &tmp_root);
  T* tmp_root_data = tmp_root.data<T>();
  phi::DenseTensor tmp_tensor;
  TensorCopy(*tensor, phi::CPUPlace(), &tmp_tensor);
  T* data = tmp_tensor.data<T>();
  for (int i = 0; i < tmp_tensor.numel(); i++) {
    tmp_root_data[i] += data[i];
  }
  TensorCopy(tmp_root, phi::CPUPlace(), root_tensor);
}
void MultiTrainer::MergeWorkerVars() {
  for (size_t i = 0; i < need_merge_var_names_.size(); i++) {
    Variable* root_var = root_scope_->FindVar(need_merge_var_names_[i]);
    if (root_var == nullptr) {
      continue;
    }
    phi::DenseTensor* root_tensor = root_var->GetMutable<phi::DenseTensor>();
#ifdef PADDLE_WITH_HETERPS
    for (int j = 0; j < thread_num_; j++) {
#else
    for (int j = 1; j < thread_num_; j++) {
#endif
      Scope* cur_thread_scope = workers_[j]->GetThreadScope();
      Variable* thread_var =
          cur_thread_scope->FindVar(need_merge_var_names_[i]);
      if (thread_var == nullptr) {
        continue;
      }
      phi::DenseTensor* thread_tensor =
          thread_var->GetMutable<phi::DenseTensor>();
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
}

void MultiTrainer::Finalize() {
  if (need_dump_field_ || need_dump_param_) {
    FinalizeDumpEnv();
  }
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
  // gpugraph mode
  if (use_gpu_graph_ && use_ps_gpu_) {
    // graph copy dense param to root scope
    for (int i = 0; i < thread_num_; ++i) {
      workers_[i]->Finalize();
    }
  } else if (use_ps_gpu_) {  // gpups mode
    MergeDenseParam();
  }
#endif
#if defined PADDLE_WITH_PSCORE
  auto communicator = paddle::distributed::Communicator::GetInstance();
  // for unittest which does not call fleet.init_worker() first
  if (communicator == nullptr) {
    VLOG(1) << "MultiTrainer::Finalize communicator is null!";
  } else {
    if (communicator->_worker_ptr != nullptr) {
      communicator->_worker_ptr->Flush();
      VLOG(1) << "MultiTrainer::Finalize ps client flush done";
    } else {
      VLOG(1) << "communicator->_worker_ptr is null";
    }
  }
#endif
  root_scope_->DropKids();
}
#ifdef PADDLE_WITH_HETERPS
void MultiTrainer::ResetDataset(Dataset* dataset) {
  SetDataset(dataset);
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();
  VLOG(3) << "readers num: " << readers.size();
  // change thread num is not supported
  PADDLE_ENFORCE_EQ(thread_num_,
                    readers.size(),
                    common::errors::InvalidArgument(
                        "change Dataset thread_num is not supported"));
  for (int i = 0; i < thread_num_; ++i) {
    workers_[i]->SetDataFeed(readers[i]);
    workers_[i]->SetPlace(places_[i]);
    workers_[i]->SetReaderPlace(places_[i]);
    workers_[i]->BindingDataFeedMemory();
  }
  // reset stat_var to 0 cause the stat_var will keep increasing
  for (size_t i = 0; i < need_merge_var_names_.size(); i++) {
    Variable* root_var = root_scope_->FindVar(need_merge_var_names_[i]);
    if (!root_var) {
      continue;
    }
    if (root_var->IsType<phi::SelectedRows>()) {
      continue;
    }
    phi::DenseTensor* root_tensor = root_var->GetMutable<phi::DenseTensor>();
    for (int j = 0; j < thread_num_; ++j) {
      Scope* cur_thread_scope = workers_[j]->GetThreadScope();
      Variable* thread_var =
          cur_thread_scope->FindVar(need_merge_var_names_[i]);
      if (thread_var == nullptr) {
        continue;
      }
      phi::DenseTensor* thread_tensor =
          thread_var->GetMutable<phi::DenseTensor>();
      auto dev_ctx_ = workers_[j]->GetDeviceContext();
#define MergeCallback2(cpp_type, proto_type)                                   \
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
      phi::DenseTensor tmp_tensor;                                             \
      TensorCopy(*thread_tensor, phi::CPUPlace(), &tmp_tensor);                \
      phi::funcs::set_constant(*dev_ctx_, thread_tensor, 0.0);                 \
    }                                                                          \
  } while (0)
      _ForEachDataType_(MergeCallback2);
    }
  }
}
#endif

}  // namespace paddle::framework
