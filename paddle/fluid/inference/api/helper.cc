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
#include "paddle/fluid/framework/custom_operator.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/api/ext/op_meta_info.h"

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

void RegisterAllCustomOperator() {
  auto &op_meta_info_map = OpMetaInfoMap::Instance();
  const auto &meta_info_map = op_meta_info_map.GetMap();
  for (auto &pair : meta_info_map) {
    const auto &all_op_kernels{framework::OperatorWithKernel::AllOpKernels()};
    if (all_op_kernels.find(pair.first) == all_op_kernels.end()) {
      framework::RegisterOperatorWithMetaInfo(pair.second);
    } else {
      LOG(INFO) << "The operator `" << pair.first
                << "` has been registered. "
                   "Therefore, we will not repeat the registration here.";
    }
  }
}

}  // namespace inference
}  // namespace paddle
