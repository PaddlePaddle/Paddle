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

#include <atomic>
#include <list>
#include <memory>
#include <mutex>
#include <thread>

#include "paddle/phi/core/enforce.h"

namespace phi {
namespace distributed {

class CommAsyncRecorder;

class CommAsyncTimeProfiler {
 public:
  struct ProfilingInfo {
    int gid;
    float time_sum;
  };

  CommAsyncTimeProfiler();

  ~CommAsyncTimeProfiler();

  static CommAsyncTimeProfiler& GetInstance() {
    static CommAsyncTimeProfiler instance;
    return instance;
  }

  void Stop();
  void AddRecorder(std::shared_ptr<CommAsyncRecorder> recorder);
  std::unordered_map<int, float> GetProfiles();

 private:
  void RecordTimeLoop();
  void UpdateProfilingInfos();
  void RegisterGroupInfo(int gid);

 private:
  std::atomic<bool> terminated_;

  std::mutex recoders_list_mutex_;
  std::mutex buffers_list_mutex_;
  std::thread loop_thread_;

  std::list<std::shared_ptr<CommAsyncRecorder>> recorders_list_;
  std::list<std::shared_ptr<CommAsyncRecorder>> buffers_list_;
  std::unordered_map<int, ProfilingInfo> profiling_infos_;
};

}  // namespace distributed
}  // namespace phi
