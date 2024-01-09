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

#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/cinn/utils/type_defs.h"
#include "paddle/pir/core/operation.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {

enum MapperType {
  OPERAND,
  ATTRIBUTE,
};

class OpMapper {
  using OprandIndexsFunction = std::function<std::vector<size_t>()>;
  using AppendAttrFunction =
      std::function<void(const ::pir::Operation& op,
                         utils::AttributeMap& attrs)>;  // NOLINT

 public:
  static OpMapper& Instance() {
    static OpMapper instance;
    return instance;
  }

  bool has(const ::pir::Operation& op, MapperType type) const {
    if (type == MapperType::OPERAND) {
      return operand_funcs_.find(op.name()) != operand_funcs_.end();
    } else if (type == MapperType::ATTRIBUTE) {
      return attr_funcs_.find(op.name()) != attr_funcs_.end();
    }
    return false;
  }

  std::vector<::pir::Value> RealOprandSources(
      const ::pir::Operation& op) const {
    CHECK(has(op, MapperType::OPERAND))
        << "Not register OprandIndexsFunction for " << op.name();
    std::vector<::pir::Value> inputs;
    for (auto idx : operand_funcs_.at(op.name())()) {
      inputs.push_back(op.operand_source(idx));
    }
    return inputs;
  }

  void AppendVariantAttrs(const ::pir::Operation& op,
                          utils::AttributeMap& attrs) const {  // NOLINT
    CHECK(has(op, MapperType::ATTRIBUTE))
        << "Not register AppendAttrFunction for " << op.name();
    attr_funcs_.at(op.name())(op, attrs);
  }

 private:
  OpMapper() { RegisterMapRules(); }
  void RegisterMapRules();

  std::unordered_map<std::string, OprandIndexsFunction> operand_funcs_;
  std::unordered_map<std::string, AppendAttrFunction> attr_funcs_;
};

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
