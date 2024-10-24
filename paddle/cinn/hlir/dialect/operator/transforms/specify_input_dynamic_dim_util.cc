// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/dialect/operator/transforms/specify_input_dynamic_dim_util.h"

#include <sys/stat.h>
#include <fstream>
#include "nlohmann/json.hpp"

using Json = nlohmann::json;

namespace cinn {
namespace dialect {
namespace ir {

namespace {

std::vector<pir::InputDynamicDimSpec> DeserializeInputDynamicDimSpecFromJson(
    const Json& json) {
  std::vector<pir::InputDynamicDimSpec> res;
  for (const auto& element : json.items()) {
    pir::InputDynamicDimSpec dim_spec;
    dim_spec.dim_name = [&]() -> std::string { return element.key(); }();
    dim_spec.input_bind = [&]() {
      const auto& value = element.value();
      std::vector<std::pair<std::string, int>> res;
      PADDLE_ENFORCE_EQ(value.contains("input_bind"),
                        true,
                        ::common::errors::InvalidArgument(
                            "input dynamic dim spec must contain input_bind"));
      for (const auto& bind_item : value["input_bind"]) {
        const auto& input_name = bind_item[0].get<std::string>();
        const auto& dim_index = bind_item[1].get<int>();
        res.emplace_back(std::make_pair(input_name, dim_index));
      }
      return res;
    }();
    dim_spec.range = [&]() {
      const auto& value = element.value();
      symbol::ConstraintsManager::Range range;
      if (value.contains("min")) {
        range.min = value["min"].get<int>();
      }
      if (value.contains("max")) {
        range.max = value["max"].get<int>();
      }
      return range;
    }();
    res.emplace_back(std::move(dim_spec));
  }
  return res;
}

bool PathExists(const std::string& path) {
  struct stat statbuf;
  if (stat(path.c_str(), &statbuf) != -1) {
    return true;
  }
  return false;
}

std::vector<pir::InputDynamicDimSpec>
DeserializeInputDynamicDimSpecFromJsonFile(std::string file_path) {
  PADDLE_ENFORCE_EQ(
      PathExists(file_path),
      true,
      ::common::errors::InvalidArgument(
          "File path for input dynamic dim spec not exists: %s.", file_path));
  std::ifstream ifs(file_path);
  PADDLE_ENFORCE_EQ(
      !ifs,
      false,
      ::common::errors::InvalidArgument(
          "File path for input dynamic dim spec fail to open for reading: %s.",
          file_path));
  Json json;
  ifs >> json;
  return DeserializeInputDynamicDimSpecFromJson(json);
}

}  // namespace

void SpecifyInputDynamicDim(
    pir::Program* program,
    const std::vector<pir::InputDynamicDimSpec>& input_dynamic_dim_spec) {
  pir::ShapeConstraintIRAnalysis& shape_analysis =
      pir::ShapeAnalysisManager::Instance().Get(program);
  shape_analysis.SetInputDynamicDimSpec(input_dynamic_dim_spec);
}

void SpecifyInputDynamicDimFromFile(pir::Program* program,
                                    std::string filepath) {
  SpecifyInputDynamicDim(program,
                         DeserializeInputDynamicDimSpecFromJsonFile(filepath));
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
