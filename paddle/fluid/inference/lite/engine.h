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

#include "lite/api/cxx_api.h"

using paddle::lite::Place;
using paddle::lite::Predictor;

namespace paddle {
namespace inference {
namespace lite {

struct EngineConfig {
  std::string model;
  std::string param;
  Place prefer_place;
  std::vector<Place> valid_places;
  std::vector<std::string> neglect_passes;
  lite_api::LiteModelType model_type{lite_api::LiteModelType::kProtobuf};
  bool memory_from_memory{true};
};

class EngineManager {
 public:
  bool Empty() const;
  bool Has(const std::string& name) const;
  Predictor* Get(const std::string& name) const;
  Predictor* Create(const std::string& name, const EngineConfig& cfg);
  void DeleteAll();
 private:
  std::unordered_map<std::string, std::unique_ptr<Predictor>> engines_;
};

}  // namespace lite
}  // namespace inference
}  // namespace paddle
