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

void PipelineTrainer::Initialize(const TrainerDesc& trainer_desc,
                                 Dataset* dataset) {
  line_num_ = trainer_desc.thread_num();

  SetDataset(dataset);
  // get filelist from trainer_desc here
  dataset->CreateReaders();
  VLOG(3) << "readers created";
  const std::vector<std::shared_ptr<paddle::framework::DataFeed>> readers =
      dataset->GetReaders();
  VLOG(3) << "readers num: " << readers.size();

  HTJ << "line num is: " << line_num_;
  HTJ << "reader size is:" << readers.size();

  pipeline_config_ = trainer_desc.pipeline_param();
  scope_queue_size_ = pipeline_config_.queue_size();
  sync_steps_ = pipeline_config_.sync_steps();
  section_num_ = pipeline_config_.section_config_size();

  HTJ << "queue_size: " << scope_queue_size_;
  HTJ << "section num: " << section_num_;
  HTJ << "sync_steps: " << sync_steps_;

  workers_.resize(section_num_);
  in_var_names_.resize(section_num_);
  out_var_names_.resize(section_num_);
  worker_count_.resize(section_num_);
  worker_count_mutex_.resize(section_num_);
  param_need_sync_.reset(new std::vector<std::string>);

  int reader_index = 0;
  for (int i = 0; i < section_num_; ++i) {
    const auto& section_config = pipeline_config_.section_config(i);
    int concurrency = section_config.concurrency();
    in_var_names_[i].reset(new std::vector<std::string>(
        section_config.section_in_var_names().begin(),
        section_config.section_in_var_names().end()));
    out_var_names_[i].reset(new std::vector<std::string>(
        section_config.section_out_var_names().begin(),
        section_config.section_out_var_names().end()));
    worker_count_[i].resize(line_num_);
    worker_count_mutex_[i].resize(line_num_);
    for (int j = 0; j < line_num_; ++j) {
      worker_count_[i][j] = new int(concurrency);
      worker_count_mutex_[i][j].reset(new std::mutex);
    }

    platform::Place place;
    workers_[i].resize(line_num_);
    for (int j = 0; j < line_num_; ++j) {
      workers_[i][j].resize(concurrency);

      switch (section_config.place()) {
        case SectionConfig::CPUPlace:
          place = platform::CPUPlace();
          break;
        case SectionConfig::CUDAPlace:
          // Note that one section has at most one GPU place in one pipeline
          place = platform::CUDAPlace(j);
          break;
        case SectionConfig::CUDAPinnedPlace:
          place = platform::CUDAPinnedPlace();
          break;
        default:
          PADDLE_ENFORCE(false, "Unkown place type in SectionConfig: %d",
                         section_config.place());
      }

      for (int k = 0; k < concurrency; ++k) {
        // workers_[i][j][k] =
        // std::dynamic_pointer_cast<paddle::framework::SectionWorker>
        workers_[i][j][k] = DeviceWorkerFactory::CreateDeviceWorker(
            trainer_desc.device_worker_name());
        auto this_worker =
            std::dynamic_pointer_cast<paddle::framework::SectionWorker>(
                workers_[i][j][k]);
        this_worker->SetSectionIndex(i);
        this_worker->SetDeviceIndex(j);
        this_worker->SetThreadIndex(k);
        this_worker->SetSectionNum(section_num_);
        this_worker->SetLineNum(line_num_);
        if (i == 0) this_worker->SetDataFeed(readers[reader_index++]);
        this_worker->SetPlace(place);
        this_worker->Initialize(trainer_desc);
      }
    }
    if (platform::is_gpu_place(place)) {
      std::shared_ptr<framework::ProgramDesc> program;
      program.reset(new ProgramDesc(section_config.program_desc()));
      for (auto& var : program->Block(0).AllVars()) {
        if (var->Persistable()) {
          param_need_sync_->emplace_back(var->Name());
        }
      }
    }
  }
  // set debug here
  SetDebug(trainer_desc.debug());
}

void PipelineTrainer::InitFirstScopeQueue(ScopeQueue* scope_queue, int line_id,
                                          const ProgramDesc& main_program) {
  for (int i = 0; i < scope_queue_size_; ++i) {
    Scope* scope = &line_scopes_[line_id]->NewScope();
    for (auto& var : main_program.Block(0).AllVars()) {
      if (!var->Persistable()) {
        auto* ptr = scope->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
      }
    }
    scope_queue->Send(scope);
  }
}

void PipelineTrainer::CopyParameters(const Scope& root_scope, int line_id) {
  for (const std::string& name : *param_need_sync_) {
    const LoDTensor& root_tensor = root_scope.FindVar(name)->Get<LoDTensor>();

    // TODO(hutxian): check a new var of the same name is created in
    // pipeline_scope
    LoDTensor* gpu_tensor =
        line_scopes_[line_id]->Var(name)->GetMutable<LoDTensor>();
    platform::Place place = platform::CUDAPlace(line_id);
    TensorCopy(*static_cast<const Tensor*>(&root_tensor), place,
               *platform::DeviceContextPool::Instance().Get(place),
               static_cast<Tensor*>(gpu_tensor));
  }
}

// call only after all resources are set in current trainer
void PipelineTrainer::InitTrainerEnv(const ProgramDesc& main_program,
                                     const platform::Place& place) {
  PADDLE_ENFORCE(root_scope_, "Null root_scope pointer");
  scope_queues_.resize(section_num_);
  line_scopes_.resize(line_num_);

  for (int i = 0; i < section_num_; ++i) {
    for (int j = 0; j < line_num_; ++j) {
      scope_queues_[i].emplace_back(new ScopeQueue(scope_queue_size_));
      if (i == 0) {
        line_scopes_[j] = &root_scope_->NewScope();
        CopyParameters(*root_scope_, j);
        InitFirstScopeQueue(scope_queues_[0].back().get(), j, main_program);
      }
    }
  }

  for (int i = 0; i < section_num_; ++i) {
    for (int j = 0; j < line_num_; ++j) {
      for (size_t k = 0; k < workers_[i][j].size(); ++k) {
        auto this_worker =
            std::dynamic_pointer_cast<paddle::framework::SectionWorker>(
                workers_[i][j][k]);
        this_worker->SetRootScope(root_scope_);
        this_worker->SetCountMutex(worker_count_mutex_[i][j].get());
        this_worker->SetWorkerCount(worker_count_[i][j]);
        this_worker->SetScopeQueue(scope_queues_[i][j].get(),
                                   (i == section_num_ - 1)
                                       ? scope_queues_[0][j].get()
                                       : scope_queues_[i + 1][j].get());
        this_worker->SetVarNames(*in_var_names_[i], *out_var_names_[i]);
        if (i != section_num_ - 1) {
          this_worker->SetNextSectionPlace(
              std::dynamic_pointer_cast<paddle::framework::SectionWorker>(
                  workers_[i + 1][j][0])
                  ->place());
        }
        this_worker->CreateDeviceResource(main_program);
      }
    }
  }

  if (line_num_ <= 1) {
    return;
  }
  // For sync functor
  std::vector<platform::Place> cuda_places;
  for (int i = 0; i < line_num_; ++i) {
    cuda_places.emplace_back(platform::CUDAPlace(i));
  }
  nccl_ctx_map_.reset(new platform::NCCLContextMap(cuda_places));
  sync_functors_.resize(line_num_);
  SyncFunctor::sync_flag_ = 0;
  SyncFunctor::line_scopes_.resize(0);
  for (int j = 0; j < line_num_; ++j) {
    SyncFunctor* sync_function = new SyncFunctor(j, line_num_, sync_steps_);
    sync_function->SetSyncParam(*param_need_sync_);
    sync_function->SetNcclCtxMap(nccl_ctx_map_.get());
    SyncFunctor::line_scopes_.push_back(this->line_scopes_[j]);
    sync_functors_[j].reset(sync_function);
  }
  for (int i = section_num_ - 1; i >= 0; --i) {
    if (SectionConfig::CUDAPlace ==
        pipeline_config_.section_config(i).place()) {
      for (int j = 0; j < line_num_; ++j) {
        for (size_t k = 0; k < workers_[i][j].size(); ++k) {
          auto this_worker =
              std::dynamic_pointer_cast<paddle::framework::SectionWorker>(
                  workers_[i][j][k]);
          this_worker->SetSyncFunctor(sync_functors_[j].get());
        }
      }
      break;
    }
  }
}

void PipelineTrainer::Run() {
  VLOG(3) << "Going to run";
  for (int i = 0; i < section_num_; ++i) {
    for (int j = 0; j < line_num_; ++j) {
      for (size_t k = 0; k < workers_[i][j].size(); ++k) {
        if (!debug_) {
          section_threads_.push_back(
              std::thread(&DeviceWorker::TrainFiles, workers_[i][j][k].get()));
        } else {
          section_threads_.push_back(std::thread(
              &DeviceWorker::TrainFilesWithProfiler, workers_[i][j][k].get()));
        }
      }
    }
  }
}

void PipelineTrainer::Finalize() {
  for (auto& th : section_threads_) {
    th.join();
  }
  for (const auto& var : *param_need_sync_) {
    auto* root_tensor = root_scope_->Var(var)->GetMutable<LoDTensor>();
    auto& thread_tensor = line_scopes_[0]->FindVar(var)->Get<LoDTensor>();
    // TODO(hutuxian): use TensorCopy instead
    int res = cudaMemcpy(root_tensor->mutable_data<float>(platform::CPUPlace()),
                         thread_tensor.data<float>(),
                         thread_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToHost);
    if (res != 0) {
      printf("cuda copy error\n");
      exit(1);
    }
  }
  dataset_ptr_->DestroyReaders();
  root_scope_->DropKids();
}

}  // end namespace framework
}  // end namespace paddle
