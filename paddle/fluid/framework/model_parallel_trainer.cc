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

void ModelParallelTrainer::Initialize(const TrainerDesc& trainer_desc,
                                      Dataset* dataset) {
  // auto device_num_ = trainer_desc.thread_num();
  // Todo: (lilong12) set device_num_ correctly
  int device_num_ = 2;
  VLOG(3) << "device num: " << device_num_;

  SetDataset(dataset);
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();
  VLOG(3) << "readers num: " << readers.size();
  // Todo: (lilong12) add assert(readers.size() == 1)
  auto reader = readers[0];
  feed_var_names_ = reader->GetUseSlotAlias();

  pipeline_config_ = trainer_desc.section_param();

  section_num_ = device_num_;
  workers_.resize(section_num_);

  for (int i = 0; i < section_num_; ++i) {
    const auto& section_config = pipeline_config_.section_config(i);
    // Todo: (lilong12) set concurrency_ correctly.
    concurrency_ = 1;
    platform::Place place;
    switch (section_config.place()) {
      case SectionConfig::CPUPlace:
        place = platform::CPUPlace();
        break;
      case SectionConfig::CUDAPlace:
        place = platform::CUDAPlace(i);
        break;
      case SectionConfig::CUDAPinnedPlace:
        place = platform::CUDAPinnedPlace();
        break;
      default:
        PADDLE_ENFORCE(false, "Unkown place type in SectionConfig: %d",
                       section_config.place());
    }
    VLOG(3) << "place: " << place;

    workers_[i].resize(concurrency_);
    for (int j = 0; j < concurrency_; ++j) {
      workers_[i][j] = DeviceWorkerFactory::CreateDeviceWorker(
          trainer_desc.device_worker_name());
      auto this_worker =
          std::dynamic_pointer_cast<paddle::framework::ModelParallelWorker>(
              workers_[i][j]);
      if (i == 0 && j == 0) {
        this_worker->SetDataFeed(reader);
        this_worker->SetReaderPlace(place);
      }
      this_worker->SetPlace(place);
      this_worker->Initialize(trainer_desc);
    }
  }
  // set debug here
  SetDebug(trainer_desc.debug());
}

// void ModelParallelTrainer::CopyParameters(const Scope& scope, int
// macrobatch_id,
void ModelParallelTrainer::CopyParameters(int macrobatch_id,
                                          const ProgramDesc& main_program) {
  auto& global_block = main_program.Block(0);
  for (auto& var : global_block.AllVars()) {
    int is_feed_var =
        std::count(feed_var_names_.begin(), feed_var_names_.end(), var->Name());
    if ((var->Persistable() || is_feed_var) && macrobatch_id == 0) {
      auto* ptr = root_scope_->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
      VLOG(3) << "Create variable " << var->Name()
              << " globally for root scope";
    } else {
      auto* ptr = macrobatch_scopes_[macrobatch_id]->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
      VLOG(3) << "Create variable " << var->Name() << " for macrobatch "
              << macrobatch_id;
    }
  }
}

void ModelParallelTrainer::InitTrainerEnv(const ProgramDesc& main_program,
                                          const platform::Place& place) {
  PADDLE_ENFORCE(root_scope_, "Null root_scope pointer");
  // Todo: (lilong12) set num_macrobatches correctly
  int num_macrobatches_ = 1;
  macrobatch_scopes_.resize(num_macrobatches_);
  for (auto& var : main_program.Block(0).AllVars()) {
    if (var->Persistable()) {
      persistable_vars_.push_back(var->Name());
    }
  }

  VLOG(3) << "create all scopes";
  // auto* scope = &root_scope_->NewScope();
  for (int i = 0; i < num_macrobatches_; ++i) {
    macrobatch_scopes_[i] = &root_scope_->NewScope();
    // CopyParameters(*root_scope_, i, main_program);
    CopyParameters(i, main_program);
  }

  for (int i = 0; i < section_num_; ++i) {
    for (size_t j = 0; j < workers_[i].size(); ++j) {
      auto this_worker =
          std::dynamic_pointer_cast<paddle::framework::ModelParallelWorker>(
              workers_[i][j]);
      this_worker->SetRootScope(root_scope_);
      this_worker->SetMacrobatchScopes(macrobatch_scopes_);
    }
  }
}

void ModelParallelTrainer::Run() {
  VLOG(3) << "Going to run";
  for (int i = 0; i < section_num_; ++i) {
    for (size_t j = 0; j < workers_[i].size(); ++j) {
      // if (!debug_) {
      if (true) {
        section_threads_.push_back(
            std::thread(&DeviceWorker::TrainFiles, workers_[i][j].get()));
      }
      // else {
      //  section_threads_.push_back(std::thread(
      //      &DeviceWorker::TrainFilesWithProfiler, workers_[i][j].get()));
      //}
      //}
    }
  }
}

void ModelParallelTrainer::Finalize() {
  VLOG(3) << "calling finalized. ";
  for (auto& th : section_threads_) {
    th.join();
  }
  root_scope_->DropKids();
}

Scope* ModelParallelTrainer::GetWorkerScope(int thread_id) {
  return macrobatch_scopes_[thread_id];
}

}  // end namespace framework
}  // end namespace paddle
