// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"

namespace paddle {
namespace framework {

class Variable;

void HeterPipelineTrainer::ResetDataset(Dataset* dataset) {
  SetDataset(dataset);
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();
  VLOG(3) << "readers num: " << readers.size();
  // TODO check thread_num_ == readers.size()
  for (int i = 0; i < thread_num_; ++i) {
    auto this_worker =
      std::dynamic_pointer_cast<paddle::framework::HeterSectionWorker>(workers_[i]);
      this_worker->SetDataFeed(readers[i]);
      this_worker->SetReaderPlace(place_);
  }
}

void HeterPipelineTrainer::Initialize(const TrainerDesc& trainer_desc,
                                      Dataset* dataset) {

  thread_num_ = trainer_desc.thread_num();
  ParseDumpConfig(trainer_desc);

  for (int i = 0; i < trainer_desc.downpour_param().stat_var_names_size();
       i++) {
    need_merge_var_names_.push_back(
        trainer_desc.downpour_param().stat_var_names(i));
  }

  // get filelist from trainer_desc here
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();
  VLOG(3) << "readers num: " << readers.size();
  // change thread num to readers num
  thread_num_ = readers.size();
  VLOG(3) << "worker thread num: " << thread_num_;
  workers_.resize(thread_num_);

  const auto& section_params = trainer_desc.section_param();
  num_pipeline_stages_ = section_params.num_pipeline_stages();
  pipeline_stage_ = section_params.pipeline_stage();
  num_microbatches_ = section_params.num_microbatches();
  VLOG(3) << "Number of microbatches per minibatch: " << num_microbatches_;
  trainer_desc_ = trainer_desc;
  trainer_id_ = trainer_desc.trainer_id();
  trainers_ = trainer_desc.trainers();

  for (int i = 0; i < thread_num_; ++i) {
    workers_[i] = DeviceWorkerFactory::CreateDeviceWorker(
        trainer_desc.device_worker_name());
    auto this_worker =
      std::dynamic_pointer_cast<paddle::framework::HeterSectionWorker>(workers_[i]);
    
    this_worker->SetNeedDumpField(need_dump_field_);
    this_worker->SetNeedDumpParam(need_dump_param_);
    this_worker->SetDumpFieldVector(dump_fields_);
    this_worker->SetDumpParamVector(dump_param_);
    this_worker->InitRandomDumpConfig(trainer_desc);

    
    //this_worker->Initialize(trainer_desc);
    this_worker->SetThreadIndex(i);
    if (pipeline_stage_ == 0) {
      this_worker->SetDataFeed(readers[i]);
    }
    this_worker->SetMicrobatchNum(num_microbatches_);
    this_worker->SetPipelineStageNum(num_pipeline_stages_);
    this_worker->SetPipelineStage(pipeline_stage_);
    this_worker->SetTrainerId(trainer_id_);
    this_worker->SetTrainers(trainers_);
    this_worker->SetThreadNum(thread_num_);
  }
  // set debug here
  SetDebug(trainer_desc.debug());
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
  // TODO(sandyhouse): should make it as a config
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
  for (int i = 0; i < thread_num_; ++i) {
    auto this_worker =
      std::dynamic_pointer_cast<paddle::framework::HeterSectionWorker>(workers_[i]);
    this_worker->SetPlace(place);
    this_worker->Initialize(trainer_desc_);
    if (pipeline_stage_ == 0) {
      this_worker->SetReaderPlace(place);
    }
    this_worker->SetRootScope(root_scope_);
    this_worker->CacheProgram(main_program);

    // generate mini_batch scope for every worker
    minibatch_scope_ = &root_scope_->NewScope();
    this_worker->SetMinibatchScope(minibatch_scope_);
    // after set micro num & mini batch scope 
    this_worker->CreateMicrobatchScopes();
  }
}

void HeterPipelineTrainer::Run() {
  VLOG(3) << "Going to run PipelineTrainer::Run()";
  if (listen_ptr_ == nullptr) {
    auto worker_0 =
      std::dynamic_pointer_cast<paddle::framework::HeterSectionWorker>(workers_[0]);
    listen_ptr_.reset(
        new std::thread(std::bind(&HeterSectionWorker::RunListen, worker_0.get())));
  } 
  VLOG(3) << "for debug " << threads_.size();
  if (pipeline_stage_ == 0) { // for cpu trainer
    for (int thidx = 0; thidx < thread_num_; ++thidx) {
        threads_.push_back(
          std::thread(&DeviceWorker::TrainFiles, workers_[thidx].get()));
    }
  } else { // for heter worker
    threads_.push_back(
        std::thread(&DeviceWorker::TrainFiles, workers_[0].get()));
    (listen_ptr_.get())->join();
  }
  for (auto& th : threads_) {
    th.join();
  }
  threads_.clear();
}

void HeterPipelineTrainer::Finalize() {
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
