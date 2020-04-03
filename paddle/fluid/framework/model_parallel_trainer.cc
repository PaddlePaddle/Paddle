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
  const auto& section_params = trainer_desc.section_param();
  // We set the blocking queue capacity to the value
  // of number of macrobatches in the python side.
  num_macrobatches_ = section_params.queue_size();
  VLOG(3) << "Number of macrobatches per minibatch: " << num_macrobatches_;
  section_num_ = section_params.section_config_size();
  VLOG(3) << "Number of training devices: " << section_num_;
  trainer_desc_ = trainer_desc;

  SetDataset(dataset);
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();
  int num_readers = readers.size();
  PADDLE_ENFORCE_EQ(num_readers, 1,
                    "Number of dataset readers for model parallel"
                    "must be 1 now, but the value you give is %d.",
                    num_readers);
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
        PADDLE_ENFORCE(false, "Unkown place type in SectionConfig: %d.",
                       section_config.place());
    }
    VLOG(3) << "Device worker place: " << place << ", device id: " << place_id
            << " for section: " << i;

    workers_[i] = DeviceWorkerFactory::CreateDeviceWorker(
        trainer_desc.device_worker_name());
    auto this_worker =
        std::dynamic_pointer_cast<paddle::framework::ModelParallelWorker>(
            workers_[i]);
    if (i == 0) {
      // for the first section
      this_worker->SetDataFeed(reader);
      this_worker->SetReaderPlace(place);
    }
    this_worker->SetThreadIndex(i);
    this_worker->SetSectionIndex(i);
    this_worker->SetPlace(place);
    this_worker->Initialize(trainer_desc);
    this_worker->SetMacrobatchNum(num_macrobatches_);
  }
  // set debug here
  SetDebug(trainer_desc.debug());
  VLOG(3) << "Initialized model parallel trainer.";
}

bool ModelParallelTrainer::isPersistableVarGrad(std::string name) {
  std::size_t pos = name.rfind(framework::kGradVarSuffix);
  if (pos == std::string::npos) {
    return false;
  }
  std::string var_name = name.substr(0, pos);
  if (persistable_var_names_.find(var_name) != persistable_var_names_.end()) {
    return true;
  }
  return false;
}

void ModelParallelTrainer::CopyParameters(int section_id, int macrobatch_id,
                                          const ProgramDesc& program,
                                          const platform::Place& place) {
  auto& global_block = program.Block(0);
  for (auto& var : global_block.AllVars()) {
    int is_feed_var =
        std::count(feed_var_names_.begin(), feed_var_names_.end(), var->Name());
    if (var->Persistable() && macrobatch_id == 0) {
      auto* ptr = root_scope_->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
      // const LoDTensor& root_tensor = ptr->Get<LoDTensor>();
      // auto* new_ptr = minibatch_scope_->Var(var->Name());
      // InitializeVariable(new_ptr, var->GetType());
      // LoDTensor* minibatch_tensor = new_ptr->GetMutable<LoDTensor>();
      // TensorCopy(*static_cast<const Tensor*>(&root_tensor), place,
      //           static_cast<Tensor*>(minibatch_tensor));
      VLOG(3) << "Create persistable var " << var->Name()
              << " for root scope, which pointer is " << ptr;
    } else if (is_feed_var && macrobatch_id == 0) {
      auto* ptr = minibatch_scope_->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
      VLOG(3) << "Create feed var " << var->Name()
              << " for minibatch scope, which pointer is " << ptr;
    } else if (isPersistableVarGrad(var->Name())) {
      auto* ptr = minibatch_scope_->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
      VLOG(3) << "Create feed var " << var->Name()
              << " for minibatch scope, which pointer is " << ptr;
    } else if (!var->Persistable() && !is_feed_var) {
      auto* ptr =
          macrobatch_scopes_[section_id][macrobatch_id]->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
      VLOG(3) << "Create variable " << var->Name() << " for section "
              << section_id << " macrobatch " << macrobatch_id
              << ", which pointer is " << ptr;
    }
  }
}

void ModelParallelTrainer::InitTrainerEnv(const ProgramDesc& main_program,
                                          const platform::Place& place) {
  PADDLE_ENFORCE_NOT_NULL(root_scope_,
                          "root_scope_ for "
                          "ModelParallelTrainer::InitTrainerEnv should not "
                          "be null.");
  // ModelParallelWorker::cpu_id_.store(pipeline_config_.start_cpu_core_id());
  macrobatch_scopes_.resize(section_num_);
  minibatch_scope_ = &root_scope_->NewScope();
  auto& global_block = main_program.Block(0);
  for (auto& var : global_block.AllVars()) {
    if (var->Persistable()) {
      persistable_var_names_.insert(var->Name());
      persistable_var_grad_names_.insert(framework::GradVarName(var->Name()));
    }
  }
  for (int i = 0; i < section_num_; ++i) {
    std::shared_ptr<framework::ProgramDesc> program;
    program.reset(new ProgramDesc(
        trainer_desc_.section_param().section_config(i).program_desc()));
    VLOG(3) << "program size for section " << i << " is: " << program->Size();
    macrobatch_scopes_[i].resize(num_macrobatches_);
    for (int j = 0; j < num_macrobatches_; ++j) {
      macrobatch_scopes_[i][j] = &minibatch_scope_->NewScope();
      CopyParameters(i, j, *program, place);
    }
  }
  VLOG(3) << "Created all scopes.";

  for (int i = 0; i < section_num_; ++i) {
    auto this_worker =
        std::dynamic_pointer_cast<paddle::framework::ModelParallelWorker>(
            workers_[i]);
    this_worker->SetRootScope(root_scope_);
    this_worker->SetMinibatchScope(minibatch_scope_);
    this_worker->SetMacrobatchScopes(macrobatch_scopes_[i]);
  }
}

void ModelParallelTrainer::Run() {
  VLOG(3) << "Going to run model parallel device worker";
  for (int i = 0; i < section_num_; ++i) {
    if (!debug_) {
      threads_.push_back(
          std::thread(&DeviceWorker::TrainFiles, workers_[i].get()));
    }
    // else {
    //  section_threads_.push_back(std::thread(
    //      &DeviceWorker::TrainFilesWithProfiler, workers_[i][j].get()));
    //}
    //}
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
  return macrobatch_scopes_[thread_id][0];
}

}  // end namespace framework
}  // end namespace paddle
