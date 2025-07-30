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
#include <map>
#include <set>
#include "paddle/pir/include/core/builtin_attribute.h"

namespace pir {
std::map<std::string, Attribute> GetOrderedOriginalAttributes(
    std::string op_name,
    const std::unordered_map<std::string, Attribute>& attributes);

class OriginalAttributesFilter {
 public:
  static OriginalAttributesFilter& Instance();

  OriginalAttributesFilter(const OriginalAttributesFilter&) = delete;
  OriginalAttributesFilter(OriginalAttributesFilter&&) = delete;
  OriginalAttributesFilter& operator=(const OriginalAttributesFilter&) = delete;

  void SetOriginalAttributesMap(
      const std::unordered_map<std::string, std::set<std::string>>&
          original_attributes_map) {
    original_attributes_map_ = original_attributes_map;
  }

 private:
  OriginalAttributesFilter() {}
  std::unordered_map<std::string, std::set<std::string>>
      original_attributes_map_;
  friend std::map<std::string, Attribute> GetOrderedOriginalAttributes(
      std::string op_name,
      const std::unordered_map<std::string, Attribute>& attributes);
};
}  // namespace pir
