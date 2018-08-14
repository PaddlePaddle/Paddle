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

/*
 * This file contains some helper functions for Inference.
 *
 * NOTE The methods here are just some helper to help debug, not guarantee to be
 * efficient.
 */

#pragma once
#include <sys/time.h>
#include <sstream>
// This header will copy to the inference_lib, to be compatible with the change
// of include path, use a short include here.
#include "paddle_inference_api.h"

namespace paddle {
namespace helper {

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

template <typename T>
std::string to_string(const std::vector<T>& x, char del = ' ') {
  std::stringstream ss;
  for (auto i : x) {
    ss << i << del;
  }
  auto res = ss.str();
  if (!res.empty()) res = res.substr(0, res.size() - 1);
  return res;
}

// Help to debug a PaddleTensor.
struct TensorSniffer {
  TensorSniffer(const PaddleTensor& x) : tensor_(x) {}

  std::string info() const {
    std::stringstream ss;
    ss << "Tensor " << tensor_.name << '\n';
    ss << " - dtype: " << dtype() << '\n';
    ss << " - shape: " << shape() << '\n';
    ss << " - lod: " << lod() << '\n';
    return ss.str();
  }

  std::string shape() const {
    std::stringstream ss;
    ss << "[";
    ss << to_string(tensor_.shape);
    ss << "]";
    return ss.str();
  }

  std::string lod() const {
    std::stringstream ss;
    ss << "[";
    for (const auto& l : tensor_.lod) {
      ss << "[";
      ss << to_string(l);
      ss << "],";
    }
    ss << "]";
    return ss.str();
  }

  std::string dtype() const {
    switch (tensor_.dtype) {
      case PaddleDType::FLOAT32:
        return "float32";
        break;
      case PaddleDType::INT64:
        return "int64";
        break;
      default:
        return "unknown";
    }
  }

 private:
  const PaddleTensor& tensor_;
};

}  // namespace helper
}  // namespace paddle
