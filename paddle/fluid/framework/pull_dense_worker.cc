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
#include <time.h>
#include "paddle/fluid/framework/device_worker.h"

namespace paddle {
namespace framework {

std::shared_ptr<PullDenseWorker> PullDenseWorker::s_instance_ = NULL;

void PullDenseWorker::Initialize(const TrainerDesc& param) {
  running_ = false;
  param_ = param.pull_dense_param();
  threshold_ = param_.threshold();
  thread_num_ = param_.device_num();
  sleep_time_ms_ = param_.sleep_time_ms();
  for (size_t i = 0; i < param_.dense_table_size(); ++i) {
    // setup dense variables for each table
    int var_num = param_.dense_table(i).dense_value_name_size();
    uint64_t tid = static_cast<uint64_t>(param_.dense_table(i).table_id());
    dense_value_names_[tid].resize(var_num);
    for (int j = 0; j < var_num; ++j) {
      dense_value_names_[tid][j] = param_.dense_table(i).dense_value_name(j);
    }
    // setup training version for each table
    training_versions_[tid].resize(thread_num_, 0);
    last_versions_[tid] = 0;
    current_version_[tid] = 0;
  }
  fleet_ptr_ = FleetWrapper::GetInstance();
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

  int MAX_FAIL_NUM = 20;
  if (pull_dense_fail_times_ > MAX_FAIL_NUM) {
    LOG(FATAL) << "Pull Dense Failed Times More Than " << MAX_FAIL_NUM
               << " Times";
    exit(-1);
  }
  status_vec->resize(0);
}

void PullDenseWorker::Stop() {
  if (running_) {
    running_ = false;
    t_.join();
  }
}

int PullDenseWorker::Start() {
  running_ = true;
  t_ = std::thread(&PullDenseWorker::Run, this);
  return 0;
}

void PullDenseWorker::Run() {
  while (running_) {
    pull_dense_status_.resize(0);
    for (size_t i = 0; i < param_.dense_table_size(); ++i) {
      uint64_t tid = static_cast<uint64_t>(param_.dense_table(i).table_id());
      if (CheckUpdateParam(tid)) {
        fleet_ptr_->PullDenseVarsAsync(
            *root_scope_, tid, dense_value_names_[tid], &pull_dense_status_);
        ResetThreadVersion(tid);
      }
    }
    if (pull_dense_status_.size() != 0) {
      Wait(&pull_dense_status_);
    }
    usleep(sleep_time_ms_ * 1000);
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
  if (current_version_[table_id] - last_versions_[table_id] < threshold_) {
    return false;
  }
  return true;
}

void PullDenseWorker::ResetThreadVersion(uint64_t table_id) {
  std::lock_guard<std::mutex> lock(mutex_for_version_);
  last_versions_[table_id] = current_version_[table_id];
}

}  // namespace framework
}  // namespace paddle
