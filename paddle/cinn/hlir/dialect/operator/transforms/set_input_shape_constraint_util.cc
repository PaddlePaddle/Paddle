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

#include "paddle/cinn/hlir/dialect/operator/transforms/set_input_shape_constraint_util.h"

#include <sys/stat.h>
#include <fstream>
#include "nlohmann/json.hpp"

using Json = nlohmann::json;

namespace cinn {
namespace dialect {
namespace ir {

namespace {

std::vector<pir::ConstraintsForInputDimName>
DeserializeInputShapeConstrainsFromJson(const Json& json) {
  std::vector<pir::ConstraintsForInputDimName> all_constraints;
  for (auto& element : json.items()) {
    pir::ConstraintsForInputDimName dim_constraint;
    dim_constraint.dim_name = [&]() -> std::string { return element.key(); }();
    dim_constraint.bind_info = [&]() {
      const auto& value = element.value();
      std::vector<std::pair<std::string, int>> res;
      PADDLE_ENFORCE_EQ(value.contains("bind_dim"),
                        true,
                        ::common::errors::InvalidArgument(
                            "input dim constriant must contain bind_dim"));
      for (const auto& bind_item : value["bind_dim"]) {
        const auto& input_name = bind_item[0].get<std::string>();
        const auto& dim_index = bind_item[1].get<int>();
        res.emplace_back(std::make_pair(input_name, dim_index));
      }
      return res;
    }();
    dim_constraint.range = [&]() {
      const auto& value = element.value();
      symbol::ConstraintsManager::Range res;
      if (value.contains("min")) {
        res.min = value["min"].get<int>();
      }
      if (value.contains("max")) {
        res.max = value["max"].get<int>();
      }
      return res;
    }();
    all_constraints.emplace_back(std::move(dim_constraint));
  }
  return all_constraints;
}

bool PathExists(const std::string& path) {
  struct stat statbuf;
  if (stat(path.c_str(), &statbuf) != -1) {
    return true;
  }
  return false;
}

std::vector<pir::ConstraintsForInputDimName>
DeserializeInputShapeConstrainsFromJsonFile(std::string file_path) {
  PADDLE_ENFORCE_EQ(
      PathExists(file_path),
      true,
      ::common::errors::InvalidArgument(
          "File path for input shape constraint not exists: %s.", file_path));
  std::ifstream ifs(file_path);
  PADDLE_ENFORCE_EQ(
      !ifs,
      false,
      ::common::errors::InvalidArgument(
          "File path for input shape constraint fail to open for reading: %s.",
          file_path));
  Json json;
  ifs >> json;
  return DeserializeInputShapeConstrainsFromJson(json);
}

}  // namespace

void SetInputShapeConstraint(
    pir::Program* program,
    const std::vector<pir::ConstraintsForInputDimName>& input_constraints) {
  pir::ShapeConstraintIRAnalysis& shape_analysis =
      pir::ShapeAnalysisManager::Instance().Get(program);
  shape_analysis.SetInputShapeConstraints(input_constraints);
}

void SetInputShapeConstraintFromFile(pir::Program* program,
                                     std::string filepath) {
  SetInputShapeConstraint(
      program, DeserializeInputShapeConstrainsFromJsonFile(filepath));
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
