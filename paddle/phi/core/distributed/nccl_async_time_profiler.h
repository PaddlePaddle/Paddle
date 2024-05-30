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
#pragma once

#include <ThreadPool.h>

#include <atomic>
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <thread>

#include "paddle/phi/core/enforce.h"

namespace phi {
namespace distributed {

class NCCLAsyncRecorder;

class NCCLAsyncTimeProfiler {
 public:
  NCCLAsyncTimeProfiler();

  ~NCCLAsyncTimeProfiler();

  static NCCLAsyncTimeProfiler& GetInstance() {
    static NCCLAsyncTimeProfiler instance;
    return instance;
  }

  void RegisterTimer(int gid, int rank, std::string recorder_name) {
    time_infos_[gid] = NCCLGroupTimeInfo();
    time_infos_[gid].rank = rank;
    time_infos_[gid].recorder_name = recorder_name;
  }
  void Stop();
  void LogSingleStep();

  void AddRecorder(std::shared_ptr<NCCLAsyncRecorder> recorder);

 private:
  void RecordTimeLoop();
  void ClearLoop();

  void InnerAddRecorder(std::shared_ptr<NCCLAsyncRecorder> recorder);
  void AddClearRecorder(std::shared_ptr<NCCLAsyncRecorder> recorder);

 private:
  std::atomic<bool> terminated_;
  ::ThreadPool pool_;
  std::future<void> add_record_task_;

  std::mutex buffer_list_mutex_;
  std::mutex recoders_list_mutex_;
  std::mutex clear_list_mutex_;

  std::condition_variable recorders_cv_;
  std::condition_variable clear_cv_;
  std::condition_variable log_cv_;

  std::thread loop_thread_;
  std::thread clear_thread_;

  std::list<std::shared_ptr<NCCLAsyncRecorder>> buffer_list_;
  std::list<std::shared_ptr<NCCLAsyncRecorder>> recorders_list_;
  std::list<std::shared_ptr<NCCLAsyncRecorder>> clear_list_;

  struct NCCLGroupTimeInfo {
    int gid;
    int rank;
    float time_sum = 0.f;
    int comm_count = 0;
    std::string recorder_name = "";
  };
  std::unordered_map<int, NCCLGroupTimeInfo> time_infos_;
};

}  // namespace distributed
}  // namespace phi
