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

const int ModelParallelTrainer::concurrency_ = 2;

void ModelParallelTrainer::Initialize(const TrainerDesc& trainer_desc,
                                      Dataset* dataset) {
  // Todo: (lilong12) set device_num_ correctly
  auto section_params = trainer_desc.section_param();
  num_macrobatches_ = section_params.queue_size();
  VLOG(3) << "number of macrobatches per minibatch: " << num_macrobatches_;
  section_num_ = section_params.section_config_size();
  VLOG(3) << "number of training devices: " << section_num_;

  SetDataset(dataset);
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();
  int num_readers = readers.size();
  PADDLE_ENFORCE_EQ(num_readers, 1,
                    "The number of dataset readers for model parallel"
                    "should be 1.");
  auto* reader = readers[0];
  feed_var_names_ = reader->GetUseSlotAlias();

  workers_.resize(section_num_);
  for (int i = 0; i < section_num_; ++i) {
    const auto& section_config = section_params.section_config(i);
    platform::Place place;
    int place_id = section_config.place_id();
    switch (section_config.place()) {
      case SectionConfig::CPUPlace:
        place = platform::CPUPlace();
        break;
      case SectionConfig::CUDAPlace:
        place = platform::CUDAPlace(place_id);
        break;
      case SectionConfig::CUDAPinnedPlace:
        place = platform::CUDAPinnedPlace();
        break;
      default:
        PADDLE_ENFORCE(false, "Unkown place type in SectionConfig: %d",
                       section_config.place());
    }
    VLOG(3) << "device worker place: " << place << ", id: " << place_id;

    workers_[i].resize(concurrency_);
    for (int j = 0; j < concurrency_; ++j) {
      VLOG(3) << "set device worker " << i << ":" << j;
      workers_[i][j] = DeviceWorkerFactory::CreateDeviceWorker(
          trainer_desc.device_worker_name());
      auto this_worker =
          std::dynamic_pointer_cast<paddle::framework::ModelParallelWorker>(
              workers_[i][j]);
      if (i == 0 && j == 0) {
        this_worker->SetDataFeed(reader);
        this_worker->SetReaderPlace(place);
      }
      int thread_index = i * concurrency_ + j;
      VLOG(3) << "set thread index: " << thread_index;
      this_worker->SetThreadIndex(thread_index);
      this_worker->SetSectionIndex(i);
      VLOG(3) << "setting place: " << place;
      this_worker->SetPlace(place);
      VLOG(3) << "setting trainer_desc";
      this_worker->Initialize(trainer_desc);
      VLOG(3) << "setting number of macrobatches";
      this_worker->SetMacrobatchNum(num_macrobatches_);
      VLOG(3) << "initialized thread: " << thread_index;
    }
  }
  VLOG(3) << "All device worker started.";
  // set debug here
  SetDebug(trainer_desc.debug());
}

void ModelParallelTrainer::CopyParameters(int macrobatch_id,
                                          const ProgramDesc& main_program) {
  auto& global_block = main_program.Block(0);
  for (auto& var : global_block.AllVars()) {
    int is_feed_var =
        std::count(feed_var_names_.begin(), feed_var_names_.end(), var->Name());
    if (var->Persistable() || is_feed_var) {
      if (macrobatch_id == 0) {
        auto* ptr = root_scope_->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
        VLOG(3) << "Create feed var " << var->Name()
                << " for root scope, which pointer is " << ptr;
      }
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
  PADDLE_ENFORCE_NOT_NULL(root_scope_,
                          "root_scope_ for "
                          "ModelParallelTrainer::InitTrainerEnv should not "
                          "be null.");
  macrobatch_scopes_.resize(num_macrobatches_);
  // auto* scope = &root_scope_->NewScope();
  for (int i = 0; i < num_macrobatches_; ++i) {
    macrobatch_scopes_[i] = &root_scope_->NewScope();
    CopyParameters(i, main_program);
  }
  VLOG(3) << "created macrobatch scopes.";

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
  VLOG(3) << "Going to run model parallel device worker";
  for (int i = 0; i < section_num_; ++i) {
    for (size_t j = 0; j < workers_[i].size(); ++j) {
      if (!debug_) {
        threads_.push_back(
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
  for (auto& th : threads_) {
    th.join();
  }
  VLOG(3) << "trainer finalized. ";
  root_scope_->DropKids();
}

Scope* ModelParallelTrainer::GetWorkerScope(int thread_id) {
  return macrobatch_scopes_[thread_id];
}

}  // end namespace framework
}  // end namespace paddle
