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

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "glog/logging.h"
#include "paddle/pir/include/core/dll_decl.h"

#pragma once

namespace paddle {
namespace translator {

using MutableAttributeInfo = std::vector<std::string>;

static constexpr char kPhiGradSuffix[] = "_grad";
static constexpr char kFluidVarGradSuffix[] = "@GRAD";

class IR_API OpNameNormalizer {
 private:
  OpNameNormalizer();  // Disallow instantiation outside of the class.
  std::unordered_map<std::string, std::string> op_name_mappings;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      op_arg_name_mappings;

  std::unordered_map<std::string,
                     std::unordered_map<std::string, MutableAttributeInfo>>
      op_mutable_attribute_infos;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      op_mutable_attributes;

 public:
  OpNameNormalizer(const OpNameNormalizer&) = delete;
  OpNameNormalizer& operator=(const OpNameNormalizer&) = delete;
  OpNameNormalizer(OpNameNormalizer&&) = delete;
  OpNameNormalizer& operator=(OpNameNormalizer&&) = delete;

  static auto& instance() {
    static OpNameNormalizer OpNameNormalizer;
    return OpNameNormalizer;
  }

  std::unordered_map<std::string, std::string> GetOpNameMappings() const {
    return op_name_mappings;
  }

  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
  GetOpArgNameMappings() const {
    return op_arg_name_mappings;
  }

  std::unordered_map<std::string,
                     std::unordered_map<std::string, MutableAttributeInfo>>
  GetOpMutableAttributeInfos() const {
    return op_mutable_attribute_infos;
  }

  std::unordered_map<std::string, std::unordered_set<std::string>>
  GetOpMutableAttributes() const {
    return op_mutable_attributes;
  }

  std::string operator[](const std::string& op_type) {
    if (op_name_mappings.find(op_type) == op_name_mappings.end()) {
      return op_type;
    }
    return op_name_mappings.at(op_type);
  }

  bool HasMutableAttribute(const std::string& op_type) {
    return (op_mutable_attributes.find(op_type) != op_mutable_attributes.end());
  }

  const std::unordered_set<std::string>* GetMutableAttributes(
      const std::string& op_type) {
    if (!HasMutableAttribute(op_type)) return nullptr;
    return &op_mutable_attributes.at(op_type);
  }

  const MutableAttributeInfo& GetMutableAttributeInfos(
      const std::string& op_type, const std::string& arg_name) {
    return op_mutable_attribute_infos.at(op_type).at(arg_name);
  }

  std::optional<std::string> GetDirectMapping(const std::string& op_type,
                                              const std::string& arg_name) {
    if (op_arg_name_mappings.find(op_type) == op_arg_name_mappings.end()) {
      return {};
    }
    auto& arg_mappings = op_arg_name_mappings[op_type];
    if (arg_mappings.find(arg_name) == arg_mappings.end()) {
      return {};
    }
    return arg_mappings.at(arg_name);
  }

  std::optional<std::string> GetGradNameMapping(const std::string& op_type,
                                                const std::string& arg_name) {
    std::string target = kPhiGradSuffix;
    std::string data = kFluidVarGradSuffix;

    size_t first_grad_pos = arg_name.find(target);
    size_t type_pos = op_type.find(target);
    std::string legacy_name = arg_name.substr(0, first_grad_pos);
    std::optional<std::string> ret =
        this->GetDirectMapping(op_type.substr(0, type_pos), legacy_name);
    if (ret) {
      legacy_name = ret.value();
    }
    legacy_name = legacy_name + arg_name.substr(first_grad_pos);
    for (size_t pos = 0;
         legacy_name.npos != (pos = legacy_name.find(target, pos));
         pos += data.length()) {
      legacy_name.replace(pos, target.length(), data);
    }
    return legacy_name;
  }

  std::string GetLegacyArgName(const std::string& op_type,
                               const std::string& arg_name) {
    if (auto ret = GetDirectMapping(op_type, arg_name)) {
      VLOG(10) << "[" << op_type << "] found " << ret.value();
      return ret.value();
    }

    bool is_grad_op = (op_type.find(kPhiGradSuffix) != std::string::npos);
    bool is_grad_arg = (arg_name.find(kPhiGradSuffix) != std::string::npos);

    if (is_grad_op && is_grad_arg) {
      if (auto ret = GetGradNameMapping(op_type, arg_name)) {
        VLOG(10) << "[" << op_type << "] found " << ret.value();
        return ret.value();
      }
    } else if (is_grad_op && !is_grad_arg) {
      // backward op using forward args: like trace_grad using forward input
      size_t type_pos = op_type.find(kPhiGradSuffix);
      if (auto ret = GetDirectMapping(op_type.substr(0, type_pos), arg_name)) {
        VLOG(10) << "[" << op_type << "] found " << ret.value();
        return ret.value();
      }
    }

    VLOG(10) << "[" << op_type << "] not found mapping for " << arg_name;
    return arg_name;
  }

  std::string GetLegacyAttrName(const std::string& op_type,
                                const std::string& arg_name) {
    size_t type_pos = op_type.find(kPhiGradSuffix);
    if (type_pos != std::string::npos) {
      std::string legacy_name =
          this->GetLegacyAttrName(op_type.substr(0, type_pos), arg_name);
      return legacy_name;
    }

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
