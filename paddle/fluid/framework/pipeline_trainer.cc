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
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"

namespace paddle {
namespace framework {

void PipelineTrainer::AsyncUpdate() {
#ifdef PADDLE_WITH_BOX_PS
  VLOG(0) << "Begin AsyncUpdate";
  std::vector<std::vector<LoDTensor>*> grad(4, nullptr);  // max package

  auto box_ptr = BoxWrapper::GetInstance();
  std::map<std::string, float> lr_map = box_ptr->GetLRMap();

  while (ps_buffer_->Receive(&grad[0])) {
    size_t merge_num = ps_buffer_->Size() + 1;
    if (merge_num > 4) {
      merge_num = 4;
    }
    for (size_t i = 1; i < merge_num; ++i) {
      ps_buffer_->Receive(&grad[i]);
    }
    AutoWRLock ps_lock(&ps_lock_);
    // VLOG(0) << "AsyncUpdate recevie grads, and begin to update param, merge "
    // << merge_num;
    for (size_t i = 0; i < async_param_list_.size() / 3; ++i) {
      LoDTensor* param_tensor = &ps_[i * 3];
      LoDTensor* mom1_tensor = &ps_[i * 3 + 1];
      LoDTensor* mom2_tensor = &ps_[i * 3 + 2];
      LoDTensor* grad_tensor = &(*grad[0])[i];
      auto len = async_param_size_[i * 3];
      float* grad_data = grad_tensor->mutable_data<float>(platform::CPUPlace());
      float* param_data =
          param_tensor->mutable_data<float>(platform::CPUPlace());
      float* mom1_data = mom1_tensor->mutable_data<float>(platform::CPUPlace());
      float* mom2_data = mom2_tensor->mutable_data<float>(platform::CPUPlace());

      // merge grad
      for (size_t k = 1; k < merge_num; ++k) {
        LoDTensor* other_grad_tensor = &(*grad[k])[i];
        float* other_grad_data =
            other_grad_tensor->mutable_data<float>(platform::CPUPlace());
        for (size_t j = 0; j < len; ++j) {
          grad_data[j] += other_grad_data[j];
        }
      }
      if (merge_num > 1) {
        for (size_t j = 0; j < len; ++j) {
          grad_data[j] /= merge_num;
        }
      }
      // float tmp = param_data[0];
      float learning_rate = base_lr_;
      if (lr_map.find(async_param_list_[i * 3]) != lr_map.end()) {
        learning_rate = lr_map[async_param_list_[i * 3]];
      }
      // VLOG(0) << "learning rate for " << async_param_list_[i * 3] << " is "
      // << learning_rate;
      for (size_t j = 0; j < len; ++j) {
        mom1_data[j] = 0.99 * mom1_data[j] +
                       0.01 * grad_data[j];  // magic beta and episilon
        mom2_data[j] =
            0.9999 * mom2_data[j] + 0.0001 * grad_data[j] * grad_data[j];
        param_data[j] -=
            learning_rate * (mom1_data[j] / (sqrt(mom2_data[j]) + 1e-8));
      }
      // VLOG(0) << "update dense for " << async_param_list_[i*3] << ", param["
      // << tmp << "] - 0.000005 * [" << mom1_data[0] << "] / [" << mom1_data[1]
      // << "] = [" << param_data[0]  << "]";
    }
  }
  VLOG(0) << "Quit AsyncUpdate";
#endif
}

void PipelineTrainer::Initialize(const TrainerDesc& trainer_desc,
                                 Dataset* dataset) {
  pipeline_num_ = trainer_desc.thread_num();
  VLOG(3) << "pipeline num: " << pipeline_num_;

  SetDataset(dataset);
  ParseDumpConfig(trainer_desc);
  // get filelist from trainer_desc here
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();
  VLOG(3) << "readers num: " << readers.size();

  pipeline_config_ = trainer_desc.section_param();
  scope_queue_size_ = pipeline_config_.queue_size();
  sync_steps_ = pipeline_config_.sync_steps();
  section_num_ = pipeline_config_.section_config_size();
  async_mode_ = pipeline_config_.async_mode();

  VLOG(3) << "scope_queue_size: " << scope_queue_size_;
  VLOG(3) << "section num: " << section_num_;
  VLOG(3) << "sync_steps: " << sync_steps_;

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
    VLOG(3) << "the thread num of each pipeline in section " << i
            << " is: " << concurrency;
    in_var_names_[i].reset(new std::vector<std::string>(
        section_config.section_in_var_names().begin(),
        section_config.section_in_var_names().end()));
    out_var_names_[i].reset(new std::vector<std::string>(
        section_config.section_out_var_names().begin(),
        section_config.section_out_var_names().end()));
    worker_count_[i].resize(pipeline_num_);
    worker_count_mutex_[i].resize(pipeline_num_);
    for (int j = 0; j < pipeline_num_; ++j) {
      worker_count_[i][j] = new int(concurrency);
      worker_count_mutex_[i][j].reset(new std::mutex);
    }

    platform::Place place;
    workers_[i].resize(pipeline_num_);
    for (int j = 0; j < pipeline_num_; ++j) {
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
        workers_[i][j][k] = DeviceWorkerFactory::CreateDeviceWorker(
            trainer_desc.device_worker_name());
        auto this_worker =
            std::dynamic_pointer_cast<paddle::framework::SectionWorker>(
                workers_[i][j][k]);
        this_worker->SetSectionIndex(i);
        this_worker->SetDeviceIndex(j);
        this_worker->SetThreadIndex(k);
        this_worker->SetSectionNum(section_num_);
        this_worker->SetPipelineNum(pipeline_num_);
        if (i == 0) {
          this_worker->SetDataFeed(readers[reader_index++]);
          this_worker->SetReaderPlace(place);
        }
        if (i == section_num_ - 1) {
          this_worker->SetNeedDumpField(need_dump_field_);
          this_worker->SetNeedDumpParam(need_dump_param_);
          this_worker->SetDumpFieldVector(dump_fields_);
          this_worker->SetDumpParamVector(dump_param_);
        }
        this_worker->SetPlace(place);
        this_worker->Initialize(trainer_desc);
        this_worker->InitRandomDumpConfig(trainer_desc);
      }
    }
  }
  param_need_sync_.reset(
      new std::vector<std::string>(pipeline_config_.param_need_sync().begin(),
                                   pipeline_config_.param_need_sync().end()));
  VLOG(3) << "param_need_sync_ have: ";
  for (const std::string& name : *param_need_sync_) {
    VLOG(3) << name;
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
  // Only set dump channel on the last section
  for (int j = 0; j < pipeline_num_; ++j) {
    for (size_t k = 0; k < workers_[section_num_ - 1][j].size(); ++k) {
      workers_[section_num_ - 1][j][k]->SetChannelWriter(queue_.get());
    }
  }
  // TODO(hutuxian): should make it as a config
  dump_thread_num_ = 1;
  for (int i = 0; i < dump_thread_num_; i++) {
    dump_thread_.push_back(
        std::thread(std::bind(&TrainerBase::DumpWork, this, i)));
  }
}

void PipelineTrainer::InitFirstScopeQueue(ScopeQueue* scope_queue,
                                          int pipeline_id,
                                          const ProgramDesc& main_program,
                                          const Scope& root_scope) {
  for (int i = 0; i < scope_queue_size_; ++i) {
    Scope* scope = &pipeline_scopes_[pipeline_id]->NewScope();
    for (auto& var : main_program.Block(0).AllVars()) {
      if (!var->Persistable()) {
        auto* ptr = scope->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
      } else {
        if (section_num_ == 1) {  // Means only one section and it must be
                                  // CUDAPlace, so copy all persistable vars to
                                  // pipeline scope
          const LoDTensor& root_tensor =
              root_scope.FindVar(var->Name())->Get<LoDTensor>();
          LoDTensor* gpu_tensor = pipeline_scopes_[pipeline_id]
                                      ->Var(var->Name())
                                      ->GetMutable<LoDTensor>();
          platform::Place place = platform::CUDAPlace(pipeline_id);
          TensorCopy(*static_cast<const Tensor*>(&root_tensor), place,
                     static_cast<Tensor*>(gpu_tensor));
        }
      }
    }
    scope_queue->Send(scope);
  }
}

void PipelineTrainer::CopyParameters(const Scope& root_scope, int pipeline_id) {
  for (const std::string& name : *param_need_sync_) {
    const LoDTensor& root_tensor = root_scope.FindVar(name)->Get<LoDTensor>();

    // TODO(hutxian): check a new var of the same name is created in
    // pipeline_scope
    LoDTensor* gpu_tensor =
        pipeline_scopes_[pipeline_id]->Var(name)->GetMutable<LoDTensor>();
    platform::Place place = platform::CUDAPlace(pipeline_id);
    TensorCopy(*static_cast<const Tensor*>(&root_tensor), place,
               static_cast<Tensor*>(gpu_tensor));
  }
}

void PipelineTrainer::InitTrainerEnv(const ProgramDesc& main_program,
                                     const platform::Place& place) {
  PADDLE_ENFORCE(root_scope_, "Null root_scope pointer");
  SectionWorker::cpu_id_.store(pipeline_config_.start_cpu_core_id());
  scope_queues_.resize(section_num_);
  pipeline_scopes_.resize(pipeline_num_);
  for (auto& var : main_program.Block(0).AllVars()) {
    if (var->Persistable()) {
      persistable_vars_.push_back(var->Name());
    }
  }

  VLOG(3) << "Init ScopeQueues and create all scopes";
  for (int i = 0; i < section_num_; ++i) {
    for (int j = 0; j < pipeline_num_; ++j) {
      scope_queues_[i].emplace_back(new ScopeQueue(scope_queue_size_));
      if (i == 0) {
        pipeline_scopes_[j] = &root_scope_->NewScope();
        CopyParameters(*root_scope_, j);
        InitFirstScopeQueue(scope_queues_[0].back().get(), j, main_program,
                            *root_scope_);
      }
    }
  }

  if (async_mode_) {
    VLOG(0) << "Begin Init For Aysnc Optimize";
    // Variable *v = root_scope_->Var("whole_param");
    // VLOG(0) << "create whole_param tensor done";
    // std::vector<std::string> input_names{"h4_param.w_0", "h5_param.w_0"};
    // std::string fused_var_name = "whole_param";
    // std::vector<std::string> output_names{"g_h4_param.w_0",
    // "g_h5_param.w_0"};
    // paddle::framework::AttributeMap attrs;
    // auto coa_op = framework::OpRegistry::CreateOp("coa_op", {{"Input",
    // {input_names}}},
    //                                                     {{"Output",
    //                                                     {output_names}},
    //                                                     {"FusedOutput",
    //                                                     {fused_var_name}}},
    //                                                     attrs);

    // VLOG(0) << "create op done";
    // coa_op->Run(*(root_scope_), platform::CPUPlace());

    for (const auto& e : *param_need_sync_) {
      if (e.find("param") != std::string::npos &&
          e.find("pow_acc") == std::string::npos) {
        VLOG(0) << "async mode choose " << e << " to update";
        async_param_list_.push_back(e);
        async_param_list_.push_back(e + "_moment1_0");
        async_param_list_.push_back(e + "_moment2_0");
      }
    }
    ps_.resize(async_param_list_.size());
    VLOG(0) << "async_param_list_.size(): " << async_param_list_.size();
    std::sort(
        async_param_list_.begin(),
        async_param_list_
            .end());  // xx_param.b_0, xx_param_moment1_0, xx_param_moment2_0
    for (size_t i = 0; i < async_param_list_.size(); ++i) {
      VLOG(0) << "begin to copy " << async_param_list_[i];
      const LoDTensor& root_tensor =
          root_scope_->FindVar(async_param_list_[i])->Get<LoDTensor>();
      VLOG(0) << "its size is " << root_tensor.numel();
      async_param_size_.push_back(root_tensor.numel());
      ps_[i].mutable_data<float>({root_tensor.numel(), 1},
                                 platform::CPUPlace());
      TensorCopy(*static_cast<const Tensor*>(&root_tensor),
                 platform::CPUPlace(), static_cast<Tensor*>(&(ps_[i])));
    }

    // Copy global lr for async mode
    for (const auto& e : persistable_vars_) {
      if (e.find("learning_rate_") != std::string::npos) {
        PADDLE_ENFORCE_LE(
            base_lr_, 0,
            platform::errors::PreconditionNotMet(
                "lr have been set, previous value: %f, current var is %s",
                base_lr_, e.c_str()));
        VLOG(0) << "begin to copy global learning rate: " << e;
        const LoDTensor& root_tensor =
            root_scope_->FindVar(e)->Get<LoDTensor>();
        const float* gpu_lr = root_tensor.data<float>();
        if (platform::is_gpu_place(root_tensor.place())) {
          cudaMemcpy(&base_lr_, gpu_lr, sizeof(float), cudaMemcpyDeviceToHost);
        } else {
          base_lr_ = *gpu_lr;
        }
      }
    }
    VLOG(0) << "base lr is " << base_lr_;
    ps_buffer_ = new operators::reader::BlockingQueue<std::vector<LoDTensor>*>(
        8 * 3);  // magic number
  }
  for (int i = 0; i < section_num_; ++i) {
    for (int j = 0; j < pipeline_num_; ++j) {
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
          // For data copy in adjacent different place
          this_worker->SetNextSectionPlace(
              std::dynamic_pointer_cast<paddle::framework::SectionWorker>(
                  workers_[i + 1][j][0])
                  ->place());
        }
        if (async_mode_) {
          this_worker->SetAsyncParamName(&async_param_list_);
          this_worker->SetAsyncParamSize(&async_param_size_);
          this_worker->SetPsBuffer(ps_buffer_);
          this_worker->SetAsyncMode(async_mode_);
          VLOG(0) << "set ps buffer for card " << j;
          this_worker->SetPs(&ps_, &ps_lock_);
        }
      }
    }
  }

  if (pipeline_num_ > 1 && sync_steps_ != -1) {
    construct_sync_functor();
  }
}

void PipelineTrainer::construct_sync_functor() {
  std::vector<platform::Place> cuda_places;
  for (int i = 0; i < pipeline_num_; ++i) {
    cuda_places.emplace_back(platform::CUDAPlace(i));
  }
  nccl_ctx_map_.reset(new platform::NCCLContextMap(cuda_places));
  sync_functors_.resize(pipeline_num_);
  SyncFunctor::sync_flag_ = 0;
  SyncFunctor::pipeline_scopes_.resize(0);

  for (int j = 0; j < pipeline_num_; ++j) {
    SyncFunctor* sync_function = new SyncFunctor(j, pipeline_num_, sync_steps_);
    sync_function->SetSyncParam(*param_need_sync_);
    sync_function->SetNcclCtxMap(nccl_ctx_map_.get());
    SyncFunctor::pipeline_scopes_.push_back(this->pipeline_scopes_[j]);
    sync_functors_[j].reset(sync_function);
  }
  for (int i = section_num_ - 1; i >= 0; --i) {
    if (SectionConfig::CUDAPlace ==
        pipeline_config_.section_config(i).place()) {
      for (int j = 0; j < pipeline_num_; ++j) {
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
    for (int j = 0; j < pipeline_num_; ++j) {
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
  if (async_mode_) {
    update_thread_ = new std::thread(&PipelineTrainer::AsyncUpdate, this);
  }
}

void PipelineTrainer::Finalize() {
  for (auto& th : section_threads_) {
    th.join();
  }
  if (async_mode_) {
    // must be after train thread, otherwise the ps_buffer_ will be closed first
    ps_buffer_->Close();
    update_thread_->join();
  }
  if (need_dump_field_) {
    FinalizeDumpEnv();
  }
  for (const auto& var : persistable_vars_) {
    auto* root_tensor = root_scope_->Var(var)->GetMutable<LoDTensor>();
    // TODO(hutuxian): Add a final all-reduce?
    const auto& thread_tensor =
        pipeline_scopes_[0]->FindVar(var)->Get<LoDTensor>();
    TensorCopySync(thread_tensor, platform::CPUPlace(), root_tensor);
  }
  if (async_mode_) {
    for (size_t i = 0; i < async_param_list_.size(); ++i) {
      VLOG(0) << "begin to copy back" << async_param_list_[i];
      auto* root_tensor =
          root_scope_->Var(async_param_list_[i])->GetMutable<LoDTensor>();
      TensorCopySync(*static_cast<const Tensor*>(&ps_[i]), platform::CPUPlace(),
                     root_tensor);
    }
  }
  root_scope_->DropKids();
}

Scope* PipelineTrainer::GetWorkerScope(int thread_id) {
  return pipeline_scopes_[thread_id];
}

}  // end namespace framework
}  // end namespace paddle
#endif
