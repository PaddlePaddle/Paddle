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
#include <string>
#include <vector>

#include "paddle/contrib/inference/paddle_inference_api.h"

namespace paddle {
namespace demo {

static void split(const std::string& str,
                  char sep,
                  std::vector<std::string>* pieces) {
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

/*
 * Get a summary of a PaddleTensor content.
 */
static std::string SummaryTensor(const PaddleTensor& tensor) {
  std::stringstream ss;
  int num_elems = tensor.data.length() / PaddleDtypeSize(tensor.dtype);

  ss << "data[:10]\t";
  switch (tensor.dtype) {
    case PaddleDType::INT64: {
      for (int i = 0; i < std::min(num_elems, 10); i++) {
        ss << static_cast<int64_t*>(tensor.data.data())[i] << " ";
      }
      break;
    }
    case PaddleDType::FLOAT32:
      for (int i = 0; i < std::min(num_elems, 10); i++) {
        ss << static_cast<float*>(tensor.data.data())[i] << " ";
      }
      break;
  }
  return ss.str();
}

}  // namespace demo
}  // namespace paddle
