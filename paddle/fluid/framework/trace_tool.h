// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <deque>
#include <mutex>
#include <condition_variable>

namespace paddle {
namespace framework {

class TraceGPUMemoTool
{
public:
	TraceGPUMemoTool(int device_id);
	~TraceGPUMemoTool() = default;
 
public:
  void Record(size_t op_idx, const std::string& op_name);
  void Pause();
  void Stop();
  size_t GetGPUMemoryInfo();

private:
  std::condition_variable cv_;
  std::mutex mutex_;
  bool is_stop_{false};
  bool is_pause_{true};
  std::string op_name_;
  size_t op_idx_;
  std::map<size_t, std::deque<size_t>>memory_info_;
  std::map<size_t, std::string>op_idx_2_op_name_;
  int device_id_;
};
}  // namespace framework
}  // namespace paddle
