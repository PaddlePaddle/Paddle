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

#if defined(PADDLE_WITH_PSCORE)
#include "paddle/fluid/distributed/service/heter_server.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"

namespace paddle {
namespace framework {

class Variable;

using MiniScope = std::unordered_map<int, Scope*>;
using MicroScope =
    std::unordered_map<int, std::shared_ptr<std::vector<Scope*>>>;
using TaskQueue =
    std::unordered_map<int, std::shared_ptr<::paddle::framework::BlockingQueue<
                                std::pair<std::string, int>>>>;

void HeterPipelineTrainer::ResetDataset(Dataset* dataset) {
  if (pipeline_stage_ == 0) {
    SetDataset(dataset);
    const std::vector<paddle::framework::DataFeed*> readers =
        dataset->GetReaders();
    VLOG(3) << "readers num: " << readers.size();
    // change thread num is not supported
    PADDLE_ENFORCE_EQ(thread_num_, readers.size(),
                      platform::errors::InvalidArgument(
                          "change Dataset thread_num is not supported"));
    int cnt = -1;
    for (auto& worker_pair : workers_) {
      cnt++;
      auto device_worker = worker_pair.second;
      auto this_worker =
          std::dynamic_pointer_cast<paddle::framework::HeterSectionWorker>(
              device_worker);
      this_worker->SetDataFeed(readers[cnt]);
      this_worker->SetReaderPlace(place_);
    }
  }
}

void HeterPipelineTrainer::Initialize(const TrainerDesc& trainer_desc,
                                      Dataset* dataset) {
  thread_num_ = trainer_desc.thread_num();
  ParseDumpConfig(trainer_desc);
  SetDebug(trainer_desc.debug());
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();
  VLOG(3) << "readers num: " << readers.size();
  // change thread num to readers num
  thread_num_ = readers.size();
  VLOG(3) << "worker thread num: " << thread_num_;
  const auto& heter_section_params = trainer_desc.heter_section_param();
  num_pipeline_stages_ = heter_section_params.num_pipeline_stages();
  pipeline_stage_ = heter_section_params.pipeline_stage();
  num_microbatches_ = heter_section_params.num_microbatches();
  VLOG(3) << "Number of microbatches per minibatch: " << num_microbatches_;
  trainer_desc_ = trainer_desc;
  trainer_id_ = trainer_desc.trainer_id();
  for (int i = 0; i < num_pipeline_stages_; ++i) {
    auto trainer_num = trainer_desc.trainers(i);
    trainers_.push_back(trainer_num);
  }
  int cpu_trainer_num = trainers_[0];
  int cur_stage_trainer_num = trainers_[pipeline_stage_];
  int global_thread_num = cpu_trainer_num * thread_num_;
  int previous_trainers = 0;
  for (int i = 0; i < pipeline_stage_; i++) previous_trainers += trainers_[i];
  int stage_trainer_id =
      trainer_id_ - previous_trainers;  // trainer id in current stage
  int cnt = -1;
  for (int i = stage_trainer_id; i < global_thread_num;
       i += cur_stage_trainer_num) {
    cnt++;
    workers_[i] = DeviceWorkerFactory::CreateDeviceWorker(
        trainer_desc.device_worker_name());
    auto this_worker =
        std::dynamic_pointer_cast<paddle::framework::HeterSectionWorker>(
            workers_[i]);
    this_worker->SetDebug(debug_);
    this_worker->SetNeedDumpField(need_dump_field_);
    this_worker->SetNeedDumpParam(need_dump_param_);
    this_worker->SetDumpFieldVector(dump_fields_);
    this_worker->SetDumpParamVector(dump_param_);
    this_worker->InitRandomDumpConfig(trainer_desc);
    this_worker->SetDeviceIndex(i);
    if (pipeline_stage_ == 0) {
      this_worker->SetDataFeed(readers[cnt]);
    }
    this_worker->SetMicrobatchNum(num_microbatches_);
    this_worker->SetPipelineStageNum(num_pipeline_stages_);
    this_worker->SetPipelineStage(pipeline_stage_);
  }
}

void HeterPipelineTrainer::InitOtherEnv(const ProgramDesc& main_program) {
  if (need_dump_field_) {
    InitDumpEnv();
  }
}

std::string HeterPipelineTrainer::GetDumpPath(int tid) {
  return string::format_string("%s/part-%05d", dump_fields_path_.c_str(), tid);
}

void HeterPipelineTrainer::InitDumpEnv() {
  queue_ = paddle::framework::MakeChannel<std::string>();
  for (int i = 0; i < thread_num_; ++i) {
    workers_[i]->SetChannelWriter(queue_.get());
  }
  dump_thread_num_ = 1;
  for (int i = 0; i < dump_thread_num_; i++) {
    dump_thread_.push_back(
        std::thread(std::bind(&TrainerBase::DumpWork, this, i)));
  }
}

void HeterPipelineTrainer::InitTrainerEnv(const ProgramDesc& main_program,
                                          const platform::Place& place) {
  place_ = place;
  PADDLE_ENFORCE_NOT_NULL(root_scope_, platform::errors::InvalidArgument(
                                           "root_scope_ can not be nullptr"));
  // initialize mini_scopes & micro_scopes
  mini_scopes_.reset(new MiniScope{});
  micro_scopes_.reset(new MicroScope{});
  task_queue_.reset(new TaskQueue{});
  for (auto& worker_pair : workers_) {
    auto worker_index = worker_pair.first;
    auto device_worker = worker_pair.second;
    auto this_worker =
        std::dynamic_pointer_cast<paddle::framework::HeterSectionWorker>(
            device_worker);
    this_worker->SetPlace(place);
    this_worker->Initialize(trainer_desc_);
    if (pipeline_stage_ == 0) {
      this_worker->SetReaderPlace(place);
    }
    this_worker->SetRootScope(root_scope_);
    // generate mini_batch scope for every worker
    auto* minibatch_scope = &root_scope_->NewScope();
    (*mini_scopes_)[worker_index] = minibatch_scope;
    this_worker->SetMinibatchScope(minibatch_scope);
    // after set micro num & mini batch scope
    this_worker->CreateMicrobatchScopes();
    (*micro_scopes_)[worker_index] = this_worker->GetMicrobatchScopes();
    (*task_queue_)[worker_index] = this_worker->GetThreadQueue();
  }
}

void HeterPipelineTrainer::Run() {
  VLOG(3) << "Going to run HeterPipelineTrainer::Run()";
  if (listen_ptr_ == nullptr) {
    for (auto& worker_pair : workers_) {
      auto& device_worker = worker_pair.second;
      auto worker_0 =
          std::dynamic_pointer_cast<paddle::framework::HeterSectionWorker>(
              device_worker);
      listen_ptr_.reset(new std::thread(
          std::bind(&HeterSectionWorker::RunListen, worker_0.get())));
      break;
    }
  }
  auto heter_server = paddle::distributed::HeterServer::GetInstance();
  heter_server->WaitServerReady();
  // heter_server->SetMiniBatchScopes(mini_scopes_);
  heter_server->SetMicroBatchScopes(micro_scopes_);
  heter_server->SetTaskQueue(task_queue_);
  // main training logic
  if (pipeline_stage_ == 0) {  // for cpu trainer
    for (auto& worker_pair : workers_) {
      auto device_worker = worker_pair.second;
      if (!debug_) {
        threads_.push_back(
            std::thread(&DeviceWorker::TrainFiles, device_worker.get()));
      } else {
        threads_.push_back(std::thread(&DeviceWorker::TrainFilesWithProfiler,
                                       device_worker.get()));
      }
    }
  } else {  // for heter worker
    for (auto& worker_pair : workers_) {
      auto device_worker = worker_pair.second;
      if (!debug_) {
        threads_.push_back(
            std::thread(&DeviceWorker::TrainFiles, device_worker.get()));
      } else {
        threads_.push_back(std::thread(&DeviceWorker::TrainFilesWithProfiler,
                                       device_worker.get()));
      }
    }
  }
  for (auto& th : threads_) {
    th.join();
  }
  if (threads_.size() > 0) {
    threads_.clear();
  }
  VLOG(3) << "Epoch Trainging done";
}

void HeterPipelineTrainer::Finalize() {
  VLOG(3) << "HeterPipelineTrainer Finalize";
  auto heter_server = paddle::distributed::HeterServer::GetInstance();
  heter_server->Stop();
  if (listen_ptr_) {
    (listen_ptr_.get())->join();
    listen_ptr_.reset(nullptr);
  }
  if (need_dump_field_) {
    FinalizeDumpEnv();
  }
  root_scope_->DropKids();
}

Scope* HeterPipelineTrainer::GetWorkerScope(int thread_id) {
  return workers_[thread_id]->GetThreadScope();
}

}  // end namespace framework
}  // end namespace paddle
#endif
