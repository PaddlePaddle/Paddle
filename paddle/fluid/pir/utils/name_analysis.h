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

#pragma once
#include <optional>
#include <set>
#include <string>
#include <vector>
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"

namespace pir {
namespace utils {
namespace name_analysis {

std::vector<std::string> GetValueAllNames(pir::Value value);
std::string GetValueFirstName(pir::Value value);
std::optional<std::string> TryGetValueFirstName(pir::Value value);
pir::Value GetParameterValueByName(const pir::Program &program,
                                   const std::string &name);
void SetValueName(pir::Value value, const std::string name);

std::map<std::string, std::string> RenameValue(Value value,
                                               const std::string &new_name,
                                               Block *block);
std::optional<std::string> GetValueInputName(pir::Value value);

std::vector<std::string> GetValueOutputNames(pir::Value value);
pir::Value GetOutputValueByName(const pir::Program &program,
                                const std::string &name);

inline bool IsOnlyOneValueName(pir::Value value) {
  std::vector<std::string> names = GetValueAllNames(value);
  if (names.size() == 1 || names.empty()) {
    return true;
  }

  std::set<std::string> names_set(names.begin(), names.end());

  if (names_set.size() == 1) {
    return true;
  } else {
    return false;
  }
}

}  // namespace name_analysis
}  // namespace utils
}  // namespace pir
