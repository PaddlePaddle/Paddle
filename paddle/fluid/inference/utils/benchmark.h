// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <fstream>
#include <iostream>
#include <string>

namespace paddle {
namespace inference {

/*
 * Helper class to calculate the performance.
 */
struct Benchmark {
  int batch_size() const { return batch_size_; }
  void SetBatchSize(int x) { batch_size_ = x; }

  int num_threads() const { return num_threads_; }
  void SetNumThreads(int x) { num_threads_ = x; }

  bool use_gpu() const { return use_gpu_; }
  void SetUseGpu() { use_gpu_ = true; }

  float latency() const { return latency_; }
  void SetLatency(float x) { latency_ = x; }

  const std::string& name() const { return name_; }
  void SetName(const std::string& name) { name_ = name; }

  std::string SerializeToString() const;
  void PersistToFile(const std::string& path) const;

 private:
  bool use_gpu_{false};
  int batch_size_{0};
  float latency_;
  int num_threads_{1};
  std::string name_;
};

}  // namespace inference
}  // namespace paddle
