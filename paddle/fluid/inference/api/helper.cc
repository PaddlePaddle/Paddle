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

#include "paddle/fluid/inference/api/helper.h"

DEFINE_bool(profile, false, "Turn on profiler for fluid");

namespace paddle {
namespace inference {

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

int Profiler::count = 0;

std::string DescribeTensor(const PaddleTensor &tensor) {
  std::stringstream os;
  os << "Tensor [" << tensor.name << "]\n";
  os << " - type: ";
  switch (tensor.dtype) {
    case PaddleDType::FLOAT32:
      os << "float32";
      break;
    case PaddleDType::INT64:
      os << "int64";
      break;
    default:
      os << "unset";
  }
  os << '\n';

  os << " - shape: " << to_string(tensor.shape) << '\n';
  os << " - lod: ";
  for (auto &l : tensor.lod) {
    os << to_string(l) << "; ";
  }
  os << "\n";
  os << " - data: ";

  int dim = std::accumulate(tensor.shape.begin(), tensor.shape.end(), 1,
                            [](int a, int b) { return a * b; });
  for (int i = 0; i < dim; i++) {
    os << static_cast<float *>(tensor.data.data())[i] << " ";
  }
  os << '\n';
  return os.str();
}

}  // namespace inference
}  // namespace paddle
