// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <unordered_map>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/engine.h"
#include "paddle/fluid/inference/utils/singleton.h"

#include "lite/api/paddle_api.h"
#include "lite/api/paddle_place.h"

namespace paddle {
namespace inference {
namespace lite {

using lite_api::PaddlePredictor;
using lite_api::CxxConfig;

class EngineManager {
 public:
  bool Empty() const { return engines_.size() == 0; }
  bool Has(const std::string& name) const {
    if (engines_.count(name) == 0) return false;
    return engines_.at(name).get() != nullptr;
  }

  PaddlePredictor* Get(const std::string& name) const {
    return engines_.at(name).get();
  }

  PaddlePredictor* Create(
      const std::string& name, const CxxConfig& config) {
    std::shared_ptr<PaddlePredictor> p = lite_api::CreatePaddlePredictor(config);
    engines_[name].reset(p);
    return p.get();
  }

  void DeleteAll() {
    for (auto& item : engines_) {
      item.second.reset(nullptr);
    }
  }

 private:
  std::unordered_map<std::string, std::shared_ptr<PaddlePredictor>> engines_;
};

}  // namespace lite
}  // namespace inference
}  // namespace paddle
