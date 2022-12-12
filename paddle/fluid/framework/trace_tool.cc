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

#include "paddle/fluid/framework/trace_tool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

#include <thread>
#include <fstream>

namespace paddle {
namespace framework {

size_t TraceGPUMemoTool::GetGPUMemoryInfo() {
  size_t avail = 0;
  size_t total = 0;
  size_t used = 0;
  cudaSetDevice(device_id_);
  cudaMemGetInfo(&avail, &total);
  used = (total - avail) / 1024 / 1024;
  return used;
}

TraceGPUMemoTool::TraceGPUMemoTool(int device_id)
{
  device_id_ = device_id;
  std::thread t([&]() {
    while(true) {
      std::unique_lock<std::mutex> lock(mutex_);
      if(is_stop_) break;
      cv_.wait(lock, [this] {return !is_pause_; });
      memory_info_[op_idx_].push_back(GetGPUMemoryInfo());
      lock.unlock();
      // std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    });
  t.detach();
}

void TraceGPUMemoTool::Stop()
{
  std::unique_lock<std::mutex> lock(mutex_);
  is_stop_ = true;
  std::cout << "memory_info_.size(): " << memory_info_.size() << std::endl;
  if(!memory_info_.size()) return;
  std::ofstream destFile("/weishengying/out.txt", std::ios::out);
  for(auto& pair : memory_info_) {
    destFile << "{op_idx: " << pair.first << " ,op_name: " << op_idx_2_op_name_[pair.first] << "\n";
    for(auto i : pair.second) {
      destFile << i << " ";
    }
    destFile << "}\n\n";
  }
  std::cout << "clear memory_info_.size(): " << memory_info_.size() << std::endl;
  memory_info_.clear();
  std::cout << "Stop end\n";
}
void TraceGPUMemoTool::Pause()
{   
    cudaDeviceSynchronize();
    std::unique_lock<std::mutex> lock(mutex_);
    is_pause_ = true;
    cv_.notify_one();
}
void TraceGPUMemoTool::Record(size_t op_idx, const std::string& op_name)
{
    std::unique_lock<std::mutex> lock(mutex_);
    is_pause_ = false;
    op_name_ = op_name;
    op_idx_ = op_idx;
    op_idx_2_op_name_[op_idx_] = op_name_;
    cv_.notify_one();
}
 
}  // namespace framework
}  // namespace paddle
