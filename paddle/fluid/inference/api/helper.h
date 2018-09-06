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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sys/time.h>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/timer.h"
#include "paddle/fluid/platform/profiler.h"

DECLARE_bool(profile);

namespace paddle {
namespace inference {

static void split(const std::string &str, char sep,
                  std::vector<std::string> *pieces) {
  pieces->clear();
  if (str.empty()) {
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}
static void split_to_float(const std::string &str, char sep,
                           std::vector<float> *fs) {
  std::vector<std::string> pieces;
  split(str, sep, &pieces);
  std::transform(pieces.begin(), pieces.end(), std::back_inserter(*fs),
                 [](const std::string &v) { return std::stof(v); });
}
static void split_to_int64(const std::string &str, char sep,
                           std::vector<int64_t> *is) {
  std::vector<std::string> pieces;
  split(str, sep, &pieces);
  std::transform(pieces.begin(), pieces.end(), std::back_inserter(*is),
                 [](const std::string &v) { return std::stoi(v); });
}
template <typename T>
std::string to_string(const std::vector<T> &vec) {
  std::stringstream ss;
  for (const auto &c : vec) {
    ss << c << " ";
  }
  return ss.str();
}
template <>
std::string to_string<std::vector<float>>(
    const std::vector<std::vector<float>> &vec);

template <>
std::string to_string<std::vector<std::vector<float>>>(
    const std::vector<std::vector<std::vector<float>>> &vec);

template <typename T>
static void TensorAssignData(PaddleTensor *tensor,
                             const std::vector<std::vector<T>> &data) {
  // Assign buffer
  int dim = std::accumulate(tensor->shape.begin(), tensor->shape.end(), 1,
                            [](int a, int b) { return a * b; });
  tensor->data.Resize(sizeof(T) * dim);
  int c = 0;
  for (const auto &f : data) {
    for (T v : f) {
      static_cast<T *>(tensor->data.data())[c++] = v;
    }
  }
}

std::string DescribeTensor(const PaddleTensor &tensor);

static void PrintTime(int batch_size, int repeat, int num_threads, int tid,
               double latency) {
  LOG(INFO) << "batch_size: " << batch_size << ", repeat: " << repeat
            << ", threads: " << num_threads << ", thread id: " << tid
            << ", latency: " << latency << "ms";
}

// Try to make the profile safer when multiple predictor is called.
// NOTE not thread safe.
// Usage:
//   Call Profiler::Start() to start profile.
//   Call profiler::Stope() to stop the profile and print the result.
struct Profiler {
  static int count;
  // Start the profile, only start when first called.
  static void Start(bool with_gpu) {
#if !defined(_WIN32)
    if (FLAGS_profile && count++ == 0) {
      LOG(WARNING) << "Profiler is actived, might affect the performance";
      LOG(INFO) << "You can turn off by set gflags '-profile false'";

      auto tracking_device = with_gpu ? platform::ProfilerState::kAll
                                      : platform::ProfilerState::kCPU;
      platform::EnableProfiler(tracking_device);
    }
#endif
  }

  // Stop the profile and print the result. Only called when last called.
  static void Stop() {
#if !defined(_WIN32)
    if (FLAGS_profile && --count == 0) {
      platform::DisableProfiler(platform::EventSortingKey::kTotal,
                                "./profile.log");
    }
#endif
  }
};

}  // namespace inference
}  // namespace paddle
