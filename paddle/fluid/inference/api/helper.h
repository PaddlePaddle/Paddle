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

#include <sys/time.h>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include "paddle/fluid/inference/api/paddle_inference_api.h"

namespace paddle {
namespace inference {

// Timer for timer
class Timer {
 public:
  double start;
  double startu;
  void tic() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    start = tp.tv_sec;
    startu = tp.tv_usec;
  }
  double toc() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    double used_time_ms =
        (tp.tv_sec - start) * 1000.0 + (tp.tv_usec - startu) / 1000.0;
    return used_time_ms;
  }
};

void split(const std::string &str, char sep, std::vector<std::string> *pieces) {
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
void split_to_float(const std::string &str, char sep, std::vector<float> *fs) {
  std::vector<std::string> pieces;
  split(str, sep, &pieces);
  std::transform(pieces.begin(), pieces.end(), std::back_inserter(*fs),
                 [](const std::string &v) { return std::stof(v); });
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
    const std::vector<std::vector<float>> &vec) {
  std::stringstream ss;
  for (const auto &piece : vec) {
    ss << to_string(piece) << "\n";
  }
  return ss.str();
}
template <>
std::string to_string<std::vector<std::vector<float>>>(
    const std::vector<std::vector<std::vector<float>>> &vec) {
  std::stringstream ss;
  for (const auto &line : vec) {
    for (const auto &rcd : line) {
      ss << to_string(rcd) << ";\t";
    }
    ss << '\n';
  }
  return ss.str();
}
// clang-format off
void TensorAssignData(PaddleTensor *tensor, const std::vector<std::vector<float>> &data) {
  // Assign buffer
  int dim = std::accumulate(tensor->shape.begin(), tensor->shape.end(), 1, [](int a, int b) { return a * b; });
  tensor->data.Resize(sizeof(float) * dim);
  int c = 0;
  for (const auto &f : data) {
    for (float v : f) { static_cast<float *>(tensor->data.data())[c++] = v; }
  }
}

}  // namespace inference
}  // namespace paddle
