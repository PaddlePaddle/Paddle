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

#include <chrono>
#include <thread>

#include "paddle/common/macros.h"
#include "paddle/phi/core/distributed/comm_async_recorder.h"
#include "paddle/phi/core/distributed/comm_async_time_profiler.h"

namespace phi {
namespace distributed {

CommAsyncTimeProfiler::CommAsyncTimeProfiler() : terminated_(false) {
  loop_thread_ = std::thread(
      &phi::distributed::CommAsyncTimeProfiler::RecordTimeLoop, this);
}

CommAsyncTimeProfiler::~CommAsyncTimeProfiler() { Stop(); }

void CommAsyncTimeProfiler::RegisterGroupInfo(int gid) {
  profiling_infos_[gid] = ProfilingInfo();
  profiling_infos_[gid].gid = gid;
}

std::unordered_map<int, float> CommAsyncTimeProfiler::GetProfiles() {
  CommAsyncRecorder::SynchronizeAllRecorders();
  std::unique_lock<std::mutex> profiling_lk(recoders_list_mutex_);
  UpdateProfilingInfos();
  std::unordered_map<int, float> ret;
  for (auto& p : profiling_infos_) {
    ret[p.first] = p.second.time_sum;
    p.second.time_sum = 0;
  }
  return ret;
}

void CommAsyncTimeProfiler::AddRecorder(
    std::shared_ptr<CommAsyncRecorder> recorder) {
  if (!terminated_.load()) {
    std::unique_lock<std::mutex> lk(buffers_list_mutex_);
    buffers_list_.push_back(std::move(recorder));
  }
}

void CommAsyncTimeProfiler::Stop() {
  terminated_.store(true);
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
}

void CommAsyncTimeProfiler::UpdateProfilingInfos() {
  for (auto iter = recorders_list_.begin(); iter != recorders_list_.end();) {
    auto recorder = *iter;
    if (!recorder->IsStart() && recorder->QueryStart()) {
      recorder->Start();
    }
    if (recorder->IsStart() && recorder->QueryEnd()) {
      float recorder_time = recorder->RecordTime();
      if (0 == profiling_infos_.count(recorder->GetGid())) {
        RegisterGroupInfo(recorder->GetGid());
      }
      profiling_infos_[recorder->GetGid()].time_sum += recorder_time;
      recorder->EventDestroy();
      iter = recorders_list_.erase(iter);
    } else {
      ++iter;
    }
  }
}

void CommAsyncTimeProfiler::RecordTimeLoop() {
  while (!terminated_.load()) {
    std::unique_lock<std::mutex> recorders_lk(recoders_list_mutex_);
    {
      std::unique_lock<std::mutex> buffers_lk(buffers_list_mutex_);
      recorders_list_.splice(recorders_list_.end(), buffers_list_);
    }

    UpdateProfilingInfos();

    recorders_lk.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
  }
}

}  // namespace distributed
}  // namespace phi
