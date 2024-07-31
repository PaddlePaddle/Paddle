// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>

#include <chrono>
#include <thread>

#include "paddle/common/macros.h"
#include "paddle/phi/core/distributed/nccl_async_recorder.h"
#include "paddle/phi/core/distributed/nccl_async_time_profiler.h"

namespace phi {
namespace distributed {

NCCLAsyncTimeProfiler::NCCLAsyncTimeProfiler() : terminated_(false), pool_(1) {
  loop_thread_ = std::thread(
      &phi::distributed::NCCLAsyncTimeProfiler::RecordTimeLoop, this);
}

NCCLAsyncTimeProfiler::~NCCLAsyncTimeProfiler() { Stop(); }

void NCCLAsyncTimeProfiler::RegisterGroupInfo(int gid) {
  time_infos_[gid] = NCCLGroupTimeInfo();
  time_infos_[gid].gid = gid;
}

std::unordered_map<int, float> NCCLAsyncTimeProfiler::GetProfiles() const {
  std::unordered_map<int, float> ret;
  for (auto p : time_infos_) {
    ret[p.first] = p.second.time_sum;
    p.second.time_sum = 0;
  }
  return ret;
}

void NCCLAsyncTimeProfiler::AddRecorder(
    std::shared_ptr<NCCLAsyncRecorder> recorder) {
  if (!terminated_.load()) {
    if (add_record_task_.valid()) {
      add_record_task_.wait();
    }
    std::unique_lock<std::mutex> lk(recoders_list_mutex_);
    recorders_list_.push_back(std::move(recorder));
    recorders_cv_.notify_one();
  }
}

void NCCLAsyncTimeProfiler::Stop() {
  terminated_.store(true);

  if (loop_thread_.joinable()) {
    recorders_cv_.notify_one();
    loop_thread_.join();
  }
  if (add_record_task_.valid()) {
    add_record_task_.wait();
  }
}

void NCCLAsyncTimeProfiler::RecordTimeLoop() {
  while (!terminated_.load()) {
    std::unique_lock<std::mutex> recorders_lk(recoders_list_mutex_);
    recorders_cv_.wait_for(recorders_lk,
                           std::chrono::milliseconds(1000),
                           [&]() -> bool { return !recorders_list_.empty(); });

    for (auto iter = recorders_list_.begin(); iter != recorders_list_.end();) {
      auto recorder = *iter;
      if (!recorder->IsStart() && recorder->QueryStart()) {
        recorder->Start();
      }
      if (recorder->IsStart() && recorder->QueryEnd()) {
        float recorder_time = recorder->RecordTime();
        if (0 == time_infos_.count(recorder->GetGid())) {
          RegisterGroupInfo(recorder->GetGid());
        }
        time_infos_[recorder->GetGid()].time_sum += recorder_time;
        recorder->EventDestroy();
        iter = recorders_list_.erase(iter);
      } else {
        ++iter;
      }
    }

    recorders_lk.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    recorders_lk.lock();
  }
}

}  // namespace distributed
}  // namespace phi
