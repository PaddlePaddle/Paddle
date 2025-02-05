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

#include "paddle/pir/include/dialect/shape/utils/original_attributes_filter.h"

namespace pir {
OriginalAttributesFilter& OriginalAttributesFilter::Instance() {
  static OriginalAttributesFilter instance;
  return instance;
}
std::map<std::string, Attribute> GetOrderedOriginalAttributes(
    std::string op_name,
    const std::unordered_map<std::string, Attribute>& attributes) {
  std::map<std::string, ::pir::Attribute> order_attributes;

  const auto& IsValidAttrName = [&](const std::string& op_name,
                                    const std::string& attr_name) -> bool {
    static const char* kOpCallStack = "op_callstack";
    static const char* kSymShapeStr = "sym_shape_str";
    static const char* kResultName = "name";
    static const char* kStopGradient = "stop_gradient";
    if (attr_name == kOpCallStack || attr_name == kSymShapeStr ||
        attr_name == kStopGradient || attr_name == kResultName)
      return false;
    const auto& original_attributes_map =
        OriginalAttributesFilter::Instance().original_attributes_map_;
    if (original_attributes_map.count(op_name) == 0) return true;
    if (original_attributes_map.at(op_name).count(attr_name) == 0) return false;
    return true;
  };

  for (const auto& [attr_name, attr_value] : attributes) {
    if (!attr_value || !IsValidAttrName(op_name, attr_name)) continue;
    order_attributes[attr_name] = attr_value;
  }
  return order_attributes;
}

}  // namespace pir
