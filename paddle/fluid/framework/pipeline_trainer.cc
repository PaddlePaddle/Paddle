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

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"

namespace paddle {
namespace framework {

void PipelineTrainer::Initialize(const TrainerDesc& trainer_desc,
                                 Dataset* dataset) {
  const auto& section_params = trainer_desc.section_param();
  num_microbatches_ = section_params.num_microbatches();
  VLOG(3) << "Number of microbatches per minibatch: " << num_microbatches_;
  section_num_ = section_params.section_config_size();
  VLOG(3) << "Number of program sections: " << section_num_;
  trainer_desc_ = trainer_desc;
  start_cpu_core_id_ = section_params.start_cpu_core_id();

  SetDataset(dataset);
  ParseDumpConfig(trainer_desc);
  // get filelist from trainer_desc here
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();
  VLOG(3) << "readers num: " << readers.size();
  int num_readers = readers.size();
  PADDLE_ENFORCE_EQ(num_readers, 1,
                    platform::errors::InvalidArgument(
                        "Number of dataset readers for pipeline "
                        "must be 1 now, but the value you give is %d.",
                        num_readers));
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
        // Note that one section has at most one GPU place in one pipeline
        PADDLE_ENFORCE_GE(
            place_id, 0,
            platform::errors::InvalidArgument(
                "The place_id value for CUDAPlace shoud be greater "
                "than or equal to 0, but the value you give is %d.",
                place_id));
        place = platform::CUDAPlace(place_id);
        break;
      case SectionConfig::CUDAPinnedPlace:
        place = platform::CUDAPinnedPlace();
        break;
      default:
        PADDLE_ENFORCE_NOT_NULL(nullptr,
                                platform::errors::InvalidArgument(
                                    "Unkown place type in SectionConfig: %d",
                                    section_config.place()));
    }
    places_.emplace_back(place);
    VLOG(3) << "Device worker place: " << place << ", device id: " << place_id
            << ", section: " << i;

    workers_[i] = DeviceWorkerFactory::CreateDeviceWorker(
        trainer_desc.device_worker_name());
    auto this_worker =
        std::dynamic_pointer_cast<paddle::framework::SectionWorker>(
            workers_[i]);
    if (i == 0) {
      // we only set reader for the first section
      this_worker->SetDataFeed(reader);
      this_worker->SetReaderPlace(place);
    }
    this_worker->SetThreadIndex(i);
    this_worker->SetSectionIndex(i);
    this_worker->SetPlace(place);
    this_worker->Initialize(trainer_desc);
    this_worker->SetMicrobatchNum(num_microbatches_);
  }
  // set debug here
  SetDebug(trainer_desc.debug());
}

void PipelineTrainer::InitOtherEnv(const ProgramDesc& main_program) {
  if (need_dump_field_) {
    InitDumpEnv();
  }
  VLOG(3) << "init other env done.";
}

std::string PipelineTrainer::GetDumpPath(int tid) {
  return string::format_string("%s/part-%05d", dump_fields_path_.c_str(), tid);
}

void PipelineTrainer::InitDumpEnv() {
  queue_ = paddle::framework::MakeChannel<std::string>();
  // TODO(sandyhouse): should make it as a config
  dump_thread_num_ = 1;
  for (int i = 0; i < dump_thread_num_; i++) {
    dump_thread_.push_back(
        std::thread(std::bind(&TrainerBase::DumpWork, this, i)));
  }
}

void PipelineTrainer::CopyParameters(int section_id, int microbatch_id,
                                     const ProgramDesc& program,
                                     const platform::Place& place) {
  auto& global_block = program.Block(0);
  for (auto& var : global_block.AllVars()) {
    int is_feed_var =
        std::count(feed_var_names_.begin(), feed_var_names_.end(), var->Name());
    if ((var->Persistable() || is_feed_var) && microbatch_id == 0) {
      if (is_feed_var) {
        auto* new_ptr = minibatch_scopes_[section_id]->Var(var->Name());
        VLOG(3) << "data name: " << var->Name() << ", ptr: " << new_ptr;
        InitializeVariable(new_ptr, var->GetType());
      } else {
        auto* ptr = root_scope_->FindVar(var->Name());
        auto* new_ptr = minibatch_scopes_[section_id]->Var(var->Name());
        VLOG(3) << "Create persistable var " << var->Name() << " for minibatch "
                << section_id << ", which pointer is " << new_ptr;
        InitializeVariable(new_ptr, var->GetType());
        const LoDTensor& root_tensor = ptr->Get<LoDTensor>();
        LoDTensor* minibatch_tensor = new_ptr->GetMutable<LoDTensor>();
        TensorCopy(*static_cast<const Tensor*>(&root_tensor), place,
                   static_cast<Tensor*>(minibatch_tensor));
      }
    } else if (!var->Persistable() && !is_feed_var) {
      auto* ptr =
          microbatch_scopes_[section_id][microbatch_id]->Var(var->Name());
      VLOG(3) << "Create variable " << var->Name() << " for section "
              << section_id << " microbatch " << microbatch_id
              << ", which pointer is " << ptr;
      InitializeVariable(ptr, var->GetType());
    }
  }
}

void PipelineTrainer::GetSkipVars(int section_id, const ProgramDesc& program) {
  auto& global_block = program.Block(0);
  for (auto& op : global_block.AllOps()) {
    if (op->Type() != "enqueue") {
      continue;
    }
    auto input_arg_names = op->InputArgumentNames();
    PADDLE_ENFORCE_EQ(input_arg_names.size(), 1,
                      platform::errors::InvalidArgument(
                          "Number of input arguments for enqueue op must be 1, "
                          "but the value is %d.",
                          input_arg_names.size()));
    std::string input_arg_name = input_arg_names[0];
    if (input_arg_name.rfind("@GRAD") != input_arg_name.size() - 5) {
      skip_vars_[section_id].emplace_back(input_arg_name);
      VLOG(3) << "add skip var name: " << input_arg_name;
    }
  }
}

void PipelineTrainer::InitTrainerEnv(const ProgramDesc& main_program,
                                     const platform::Place& place) {
  PADDLE_ENFORCE_NOT_NULL(root_scope_,
                          platform::errors::InvalidArgument(
                              "root_scope pointer can not be nullptr"));
  auto start_cpu_id = trainer_desc_.section_param().start_cpu_core_id();
  SectionWorker::cpu_id_.store(start_cpu_id);
  minibatch_scopes_.resize(section_num_);
  microbatch_scopes_.resize(section_num_);
  skip_vars_.resize(section_num_);

  VLOG(3) << "Init ScopeQueues and create all scopes";
  for (int i = 0; i < section_num_; ++i) {
    minibatch_scopes_[i] = &root_scope_->NewScope();
    std::shared_ptr<framework::ProgramDesc> program;
    program.reset(new ProgramDesc(
        trainer_desc_.section_param().section_config(i).program_desc()));
    microbatch_scopes_[i].resize(num_microbatches_);
    for (int j = 0; j < num_microbatches_; ++j) {
      microbatch_scopes_[i][j] = &minibatch_scopes_[i]->NewScope();
      CopyParameters(i, j, *program, places_[i]);
    }
    GetSkipVars(i, *program);
  }

  for (int i = 0; i < section_num_; ++i) {
    auto this_worker =
        std::dynamic_pointer_cast<paddle::framework::SectionWorker>(
            workers_[i]);
    this_worker->SetRootScope(root_scope_);
    this_worker->SetMinibatchScope(minibatch_scopes_[i]);
    this_worker->SetMicrobatchScopes(microbatch_scopes_[i]);
    this_worker->SetSkipVars(skip_vars_[i]);
  }
}

void PipelineTrainer::Run() {
  VLOG(3) << "Going to run";
  for (int i = 0; i < section_num_; ++i) {
    if (!debug_) {
      section_threads_.push_back(
          std::thread(&DeviceWorker::TrainFiles, workers_[i].get()));
    } else {
      section_threads_.push_back(std::thread(
          &DeviceWorker::TrainFilesWithProfiler, workers_[i].get()));
    }
  }
}

void PipelineTrainer::Finalize() {
  for (auto& th : section_threads_) {
    th.join();
  }
  if (need_dump_field_) {
    FinalizeDumpEnv();
  }
  VLOG(3) << "copying back parameters. ";
  for (int i = 0; i < section_num_; ++i) {
    std::shared_ptr<framework::ProgramDesc> program;
    program.reset(new ProgramDesc(
        trainer_desc_.section_param().section_config(i).program_desc()));
    for (int j = 0; j < num_microbatches_; ++j) {
      auto& global_block = program->Block(0);
      for (auto& var : global_block.AllVars()) {
        if (var->Persistable()) {
          auto* ptr = root_scope_->FindVar(var->Name());
          LoDTensor* root_tensor = ptr->GetMutable<LoDTensor>();
          auto* minibatch_ptr = minibatch_scopes_[i]->Var(var->Name());
          const LoDTensor& minibatch_tensor = minibatch_ptr->Get<LoDTensor>();
          TensorCopy(*static_cast<const Tensor*>(&minibatch_tensor), places_[0],
                     static_cast<Tensor*>(root_tensor));
          VLOG(4) << "Copy persitable var " << var->Name() << " to root scope";
        }
      }
    }
  }
  root_scope_->DropKids();
}

Scope* PipelineTrainer::GetWorkerScope(int thread_id) {
  return microbatch_scopes_[thread_id][0];
}

}  // end namespace framework
}  // end namespace paddle
#endif
