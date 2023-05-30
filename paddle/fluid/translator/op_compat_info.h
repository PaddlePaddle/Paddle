// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <unordered_map>

#include "glog/logging.h"

#include "paddle/fluid/translator/utils.h"

#pragma once

namespace paddle {
namespace translator {

class OpNameNormalizer {
 private:
  OpNameNormalizer();  // Disallow instantiation outside of the class.
  std::unordered_map<std::string, std::string> op_name_mappings;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      op_arg_name_mappings;

 public:
  OpNameNormalizer(const OpNameNormalizer&) = delete;
  OpNameNormalizer& operator=(const OpNameNormalizer&) = delete;
  OpNameNormalizer(OpNameNormalizer&&) = delete;
  OpNameNormalizer& operator=(OpNameNormalizer&&) = delete;

  static auto& instance() {
    static OpNameNormalizer OpNameNormalizer;
    return OpNameNormalizer;
  }

  std::string operator[](const std::string& op_type) {
    if (op_name_mappings.find(op_type) == op_name_mappings.end()) {
      return op_type;
    }
    return op_name_mappings.at(op_type);
  }

  std::string GetLegacyArgName(const std::string& op_type,
                               const std::string& arg_name) {
    if (op_arg_name_mappings.find(op_type) == op_arg_name_mappings.end()) {
      return UnderscoreToCamelCase(arg_name);
    }
    auto& arg_mappings = op_arg_name_mappings[op_type];
    if (arg_mappings.find(arg_name) == arg_mappings.end()) {
      return UnderscoreToCamelCase(arg_name);
    }
    return arg_mappings.at(arg_name);
  }

  std::string GetLegacyAttrName(const std::string& op_type,
                                const std::string& arg_name) {
    if (op_arg_name_mappings.find(op_type) == op_arg_name_mappings.end()) {
      VLOG(10) << "[" << op_type << "] not found";
      return arg_name;
    }
    auto& arg_mappings = op_arg_name_mappings[op_type];
    if (arg_mappings.find(arg_name) == arg_mappings.end()) {
      VLOG(10) << "[" << op_type << "][" << arg_name << "] not found";
      return arg_name;
    }
    return arg_mappings.at(arg_name);
  }
};

}  // namespace translator
}  // namespace paddle
