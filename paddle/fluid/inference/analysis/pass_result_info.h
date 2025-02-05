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

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/phi/core/enforce.h"
#include "paddle/utils/variant.h"

namespace paddle {
namespace inference {
namespace analysis {

class PassResultInfoForRuntime {
 public:
  using PassInfo =
      paddle::variant<std::string,
                      std::vector<std::string>,
                      std::unordered_map<std::string, std::string>>;

  static PassResultInfoForRuntime* Instance() {
    static PassResultInfoForRuntime info;
    return &info;
  }

  template <typename T>
  void Set(int predictor_id, const std::string& pass_name, T infos) {
    map[predictor_id].emplace(pass_name, infos);
  }

  template <typename T>
  T Get(int predictor_id, const std::string& pass_name) {
    PADDLE_ENFORCE_EQ(
        map.count(predictor_id) && map[predictor_id].count(pass_name),
        true,
        common::errors::InvalidArgument(
            "Not find predictor_id %d and pass_name %s",
            predictor_id,
            pass_name));
    return PADDLE_GET_CONST(T, map[predictor_id][pass_name]);
  }

 private:
  using PassResultInfoMap =
      std::unordered_map<int, std::unordered_map<std::string, PassInfo>>;
  PassResultInfoMap map;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
