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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "lite/api/cxx_api.h"
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_place.h"
#include "lite/api/paddle_use_passes.h"
#pragma GCC diagnostic pop

namespace paddle {
namespace inference {
namespace lite {

struct EngineConfig {
  std::string model;
  std::string param;
  std::vector<paddle::lite_api::Place> valid_places;
  std::vector<std::string> neglected_passes;
  lite_api::LiteModelType model_type{lite_api::LiteModelType::kProtobuf};
  bool model_from_memory{true};

  // for xpu
  size_t xpu_l3_workspace_size;

  // for x86 or arm
  int cpu_math_library_num_threads{1};

  // for cuda
  bool use_multi_stream{false};
};

class EngineManager {
 public:
  bool Empty() const;
  bool Has(const std::string& name) const;
  paddle::lite_api::PaddlePredictor* Get(const std::string& name) const;
  paddle::lite_api::PaddlePredictor* Create(const std::string& name,
                                            const EngineConfig& cfg);
  void DeleteAll();

 private:
  std::unordered_map<std::string,
                     std::shared_ptr<paddle::lite_api::PaddlePredictor>>
      engines_;
};

}  // namespace lite
}  // namespace inference
}  // namespace paddle
