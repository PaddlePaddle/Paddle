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

#include "paddle/common/macros.h"
#include "paddle/phi/core/distributed/nccl_async_recorder.h"
#include "paddle/phi/core/distributed/nccl_async_time_profiler.h"

namespace phi {
namespace distributed {

NCCLAsyncTimeProfiler::NCCLAsyncTimeProfiler() : terminated_(false), pool_(1) {
  loop_thread_ = std::thread(
      &phi::distributed::NCCLAsyncTimeProfiler::RecordTimeLoop, this);

  clear_thread_ =
      std::thread(&phi::distributed::NCCLAsyncTimeProfiler::ClearLoop, this);
}

NCCLAsyncTimeProfiler::~NCCLAsyncTimeProfiler() { Stop(); }

void NCCLAsyncTimeProfiler::LogSingleStep() {
  if (!terminated_.load()) {
    // // #ifdef PADDLE_WITH_CUDA
    // //     PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
    // // #else  // PADDLE_WITH_HIP
    // //     PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
    // // #endif
    std::unique_lock<std::mutex> lk(recoders_list_mutex_);
    log_cv_.wait_for(lk, std::chrono::seconds(100), [&]() -> bool {
      return recorders_list_.empty();
    });
    if (!recorders_list_.empty()) {
      LOG(WARNING)
          << "The communication has not completed after more than 100 seconds";
    }
    for (auto& p : time_infos_) {
      LOG(INFO) << "gid:" << p.first
                << "   comm_group:" << p.second.recorder_name
                << "   comm_count:" << p.second.comm_count
                << "   time:" << p.second.time_sum
                << "   rank:" << p.second.rank;
      p.second.time_sum = 0.f;
      p.second.comm_count = 0;
    }
  }
}

void NCCLAsyncTimeProfiler::InnerAddRecorder(
    std::shared_ptr<NCCLAsyncRecorder> recorder) {
  std::unique_lock<std::mutex> lk(buffer_list_mutex_);
  buffer_list_.push_back(std::move(recorder));
  recorders_cv_.notify_one();
}

void NCCLAsyncTimeProfiler::AddRecorder(
    std::shared_ptr<NCCLAsyncRecorder> recorder) {
  if (!terminated_.load()) {
    if (add_record_task_.valid()) {
      add_record_task_.wait();
    }
    add_record_task_ = pool_.enqueue(
        &NCCLAsyncTimeProfiler::InnerAddRecorder, this, std::move(recorder));
  }
}

void NCCLAsyncTimeProfiler::AddClearRecorder(
    std::shared_ptr<NCCLAsyncRecorder> recorder) {
  if (!terminated_.load()) {
    std::unique_lock<std::mutex> lk(clear_list_mutex_);
    clear_list_.push_back(std::move(recorder));
    clear_cv_.notify_one();
  }
}

void NCCLAsyncTimeProfiler::Stop() {
  terminated_.store(true);

  if (clear_thread_.joinable()) {
    clear_cv_.notify_one();
    clear_thread_.join();
  }
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
    recorders_cv_.wait_for(
        recorders_lk, std::chrono::milliseconds(1000), [&]() -> bool {
          return !recorders_list_.empty() || !buffer_list_.empty();
        });

    {
      std::unique_lock<std::mutex> buffer_lk(buffer_list_mutex_);
      recorders_list_.splice(recorders_list_.end(), buffer_list_);
    }

    recorders_lk.unlock();
    for (auto iter = recorders_list_.begin(); iter != recorders_list_.end();) {
      auto recorder = *iter;
      if (!recorder->IsStart() && recorder->QueryStart()) {
        recorder->Start();
      }
      if (recorder->IsStart() && recorder->QueryEnd()) {
        float recorder_time = recorder->RecordTime();
        time_infos_[recorder->GetGid()].time_sum += recorder_time;
        time_infos_[recorder->GetGid()].comm_count++;
        AddClearRecorder(recorder);
        recorders_lk.lock();
        iter = recorders_list_.erase(iter);
        recorders_lk.unlock();
        log_cv_.notify_one();
      } else {
        ++iter;
      }
    }
  }
}

void NCCLAsyncTimeProfiler::ClearLoop() {
  while (!terminated_.load()) {
    std::unique_lock<std::mutex> lk(clear_list_mutex_);
    clear_cv_.wait_for(lk, std::chrono::milliseconds(1000), [&]() -> bool {
      return !clear_list_.empty();
    });
    for (auto iter = clear_list_.begin(); iter != clear_list_.end();) {
      auto recorder = *iter;
      recorder->EventDestroy();
      iter = clear_list_.erase(iter);
    }
  }
}

}  // namespace distributed
}  // namespace phi
