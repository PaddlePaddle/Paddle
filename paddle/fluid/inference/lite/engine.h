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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "lite/api/cxx_api.h"

namespace paddle {
namespace inference {
namespace lite {

struct EngineConfig {
  std::string model;
  std::string param;
  paddle::lite::Place prefer_place;
  std::vector<paddle::lite::Place> valid_places;
  std::vector<std::string> neglected_passes;
  lite_api::LiteModelType model_type{lite_api::LiteModelType::kProtobuf};
  bool model_from_memory{true};
};

class EngineManager {
 public:
  bool Empty() const;
  bool Has(const std::string& name) const;
  paddle::lite::Predictor* Get(const std::string& name) const;
  paddle::lite::Predictor* Create(const std::string& name,
                                  const EngineConfig& cfg);
  void DeleteAll();

 private:
  std::unordered_map<std::string, std::unique_ptr<paddle::lite::Predictor>>
      engines_;
};

}  // namespace lite
}  // namespace inference
}  // namespace paddle
