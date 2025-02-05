/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <ctime>

#include "paddle/fluid/framework/device_worker.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle::framework {

class Scope;
class Variable;

std::shared_ptr<PullDenseWorker> PullDenseWorker::s_instance_ = NULL;
std::mutex PullDenseWorker::mutex_for_version_;
std::map<uint64_t, uint64_t> PullDenseWorker::last_versions_;
std::map<uint64_t, uint64_t> PullDenseWorker::current_version_;
std::map<uint64_t, std::vector<uint64_t>> PullDenseWorker::training_versions_;
std::map<uint64_t, std::vector<std::string>>
    PullDenseWorker::dense_value_names_;

void PullDenseWorker::Initialize(const TrainerDesc& param) {
  running_ = false;
  param_ = param.pull_dense_param();
  dwp_param_ = param.downpour_param();
  threshold_ = param_.threshold();
  thread_num_ = param_.device_num();
  sleep_time_ms_ = param_.sleep_time_ms();
  for (int i = 0; i < dwp_param_.program_config(0).pull_dense_table_id_size();
       ++i) {
    uint64_t tid = static_cast<uint64_t>(
        dwp_param_.program_config(0).pull_dense_table_id(i));
    TableParameter table;
    for (auto const& i : param_.dense_table()) {
      if (i.table_id() == tid) {
        table = i;
        break;
      }
    }
    // setup dense variables for each table
    int var_num = table.dense_value_name_size();
    dense_value_names_[tid].resize(var_num);
    for (int j = 0; j < var_num; ++j) {
      dense_value_names_[tid][j] = table.dense_value_name(j);
    }
    // setup training version for each table
    training_versions_[tid].resize(thread_num_, 0);
    last_versions_[tid] = 0;
    current_version_[tid] = 0;
  }

#if defined(PADDLE_WITH_PSCORE)
  fleet_ptr_ = paddle::distributed::FleetWrapper::GetInstance();
#else
  fleet_ptr_ = FleetWrapper::GetInstance();
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  copy_streams_.clear();
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_XPU)
  places_.clear();
  thread_scopes_.clear();
#endif
}

void PullDenseWorker::CreatePinVar() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_XPU)
  // for (auto& v : dense_value_names_) {
  //  for (auto& name : v.second) {
  for (int i = 0; i < dwp_param_.program_config(0).pull_dense_table_id_size();
       ++i) {
    uint64_t tid = static_cast<uint64_t>(
        dwp_param_.program_config(0).pull_dense_table_id(i));
    for (auto& name : dense_value_names_[tid]) {
      Variable* var = root_scope_->FindVar(name);

      phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
      auto* ptr = root_scope_->Var(name + "pin");
      InitializeVariable(ptr, proto::VarType::DENSE_TENSOR);
      phi::DenseTensor* pin_tensor = ptr->GetMutable<phi::DenseTensor>();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      pin_tensor->mutable_data<float>(tensor->dims(), phi::GPUPinnedPlace());
#endif
#ifdef PADDLE_WITH_XPU
      pin_tensor->mutable_data<float>(tensor->dims(), phi::CPUPlace());
#endif
    }
  }
#endif
}

void PullDenseWorker::Wait(std::vector<::std::future<int32_t>>* status_vec) {
  for (auto& t : *status_vec) {
    t.wait();
    auto status = t.get();
    if (status != 0) {
      LOG(WARNING) << "Current Pull Dense Thread Failed Times"
                   << ++pull_dense_fail_times_;
    }
  }

  size_t MAX_FAIL_NUM = 20;
  if (pull_dense_fail_times_ > MAX_FAIL_NUM) {
    PADDLE_THROW(common::errors::Fatal("Pull dense failed more than %d times.",
                                       MAX_FAIL_NUM));
    exit(-1);
  }
  status_vec->resize(0);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_XPU)

  for (size_t i = 0; i < places_.size(); ++i) {
    // for (auto& v : dense_value_names_) {
    //  for (auto& name : v.second) {
    for (int x = 0; x < dwp_param_.program_config(0).pull_dense_table_id_size();
         ++x) {
      uint64_t tid = static_cast<uint64_t>(
          dwp_param_.program_config(0).pull_dense_table_id(x));
      for (auto& name : dense_value_names_[tid]) {
        Variable* pin_var = root_scope_->FindVar(name + "pin");
        phi::DenseTensor* pin_tensor = pin_var->GetMutable<phi::DenseTensor>();
        float* pin_w = pin_tensor->data<float>();
        Variable* var = thread_scopes_[i]->FindVar(name);
        phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
        float* w = tensor->data<float>();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        memory::Copy(places_[i],
                     w,
                     phi::GPUPinnedPlace(),
                     pin_w,
                     sizeof(float) * tensor->numel(),
                     copy_streams_[i]);
#endif
#ifdef PADDLE_WITH_XPU
        memory::Copy(places_[i],
                     w,
                     phi::CPUPlace(),
                     pin_w,
                     sizeof(float) * tensor->numel());
#endif
      }
    }
  }
#endif
}

void PullDenseWorker::Stop() {
  if (running_) {
    running_ = false;
    t_.join();
  }
}

void PullDenseWorker::PullDense(bool force_update) {
  pull_dense_status_.resize(0);
  for (int i = 0; i < dwp_param_.program_config(0).pull_dense_table_id_size();
       ++i) {
    uint64_t tid = static_cast<uint64_t>(
        dwp_param_.program_config(0).pull_dense_table_id(i));
    if (force_update || CheckUpdateParam(tid)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_XPU)
      VLOG(3) << "pull dense " << force_update << " " << tid;
      fleet_ptr_->PullDenseVarsAsync(*root_scope_,
                                     tid,
                                     dense_value_names_[tid],
                                     &pull_dense_status_,
                                     false);
#elif defined(PADDLE_WITH_PSCORE)
      fleet_ptr_->PullDenseVarsAsync(*root_scope_,
                                     tid,
                                     dense_value_names_[tid],
                                     &pull_dense_status_,
                                     true);
#else
      fleet_ptr_->PullDenseVarsAsync(*root_scope_,
                                     tid,
                                     dense_value_names_[tid],
                                     &pull_dense_status_,
                                     true);
#endif
      ResetThreadVersion(tid);
    }
  }
  if (!pull_dense_status_.empty()) {
    Wait(&pull_dense_status_);
  }
}

int PullDenseWorker::Start() {
  running_ = true;
  // before training, we can pull dense from pserver first.
  PullDense(true);
  t_ = std::thread(&PullDenseWorker::Run, this);
  return 0;
}

void PullDenseWorker::Run() {
  while (running_) {
    PullDense(false);
#ifndef _WIN32
    usleep(sleep_time_ms_ * 1000);
#endif
  }
}

void PullDenseWorker::IncreaseThreadVersion(int thread_id, uint64_t table_id) {
  std::lock_guard<std::mutex> lock(mutex_for_version_);
  training_versions_[table_id][thread_id]++;
}

bool PullDenseWorker::CheckUpdateParam(uint64_t table_id) {
  std::lock_guard<std::mutex> lock(mutex_for_version_);
  auto& version = training_versions_[table_id];
  current_version_[table_id] =
      *(std::min_element(version.begin(), version.end()));
  if (current_version_[table_id] - last_versions_[table_id] <
      static_cast<size_t>(threshold_)) {
    return false;
  }
  return true;
}

void PullDenseWorker::ResetThreadVersion(uint64_t table_id) {
  std::lock_guard<std::mutex> lock(mutex_for_version_);
  last_versions_[table_id] = current_version_[table_id];
}

int PullDenseWorker::GetThreadIdByScope(const Scope* scope) {
  if (scope_to_thread_id_.find(scope) != scope_to_thread_id_.end()) {
    return scope_to_thread_id_[scope];
  }
  return -1;
}

void PullDenseWorker::SetThreadIdByScope(const Scope* scope, int tid) {
  scope_to_thread_id_[scope] = tid;
}

void PullDenseWorker::MergeDenseParam() {
  for (int x = 0; x < dwp_param_.program_config(0).pull_dense_table_id_size();
       ++x) {
    uint64_t tid = static_cast<uint64_t>(
        dwp_param_.program_config(0).pull_dense_table_id(x));
    for (auto& name : dense_value_names_[tid]) {
      Variable* root_var = root_scope_->FindVar(name);
      phi::DenseTensor* root_tensor = root_var->GetMutable<phi::DenseTensor>();
      Variable* var = thread_scopes_[0]->FindVar(name);
      phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
      TensorCopy((*tensor), root_tensor->place(), root_tensor);
    }
  }
}

}  // namespace paddle::framework
