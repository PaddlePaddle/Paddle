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

#include "paddle/cinn/hlir/framework/new_ir/utils.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace newir {

const std::unordered_map<std::string, std::string> CompatibleInfo::OP_NAMES = {
    {"pd.full", "fill_constant"}};

std::string CompatibleInfo::OpName(const ::ir::Operation& op) {
  std::string name = op.name();
  if (OP_NAMES.count(name)) {
    return OP_NAMES.at(name);
  }
  auto pos = name.find(".");
  if (pos == std::string::npos) {
    return name;
  }
  auto cinn_op_name = name.substr(pos + 1);
  VLOG(4) << "GetOpName: " << name << " -> " << cinn_op_name;
  return cinn_op_name;
}

std::string CompatibleInfo::InputName(const ::ir::Value& value) {
  return CompatibleInfo::kInputPrefix +
         std::to_string(std::hash<::ir::Value>()(value));
}

std::string CompatibleInfo::OutputName(const ::ir::Value& value) {
  return CompatibleInfo::kOutputPrefix +
         std::to_string(std::hash<::ir::Value>()(value));
}

std::string CompatibleInfo::OpFuncName(const ::ir::Operation& op) {
  std::string op_name = OpName(op);
  std::string func_name =
      cinn::common::Context::Global().NewName("fn_" + op_name);
  return func_name;
}

std::string CompatibleInfo::GroupOpsName(
    const std::vector<::ir::Operation*>& ops) {
  std::string name = "fn_";
  for (auto* op : ops) {
    std::string op_name = OpName(*op);
    name += cinn::common::Context::Global().NewName(op_name);
  }
  return name;
}

std::vector<std::string> CompatibleInfo::InputNames(const ::ir::Operation& op,
                                                    bool allow_duplicate) {
  std::vector<std::string> names;
  std::unordered_set<std::string> repeat;
  for (int i = 0; i < op.num_operands(); ++i) {
    auto value = op.operand_source(i);
    std::string name = CompatibleInfo::InputName(value);
    if (!allow_duplicate && repeat.count(name)) {
      continue;
    }
    repeat.insert(name);
    names.push_back(name);
  }
  return names;
}

std::vector<std::string> CompatibleInfo::OutputNames(
    const ::ir::Operation& op) {
  std::vector<std::string> names;
  for (int i = 0; i < op.num_results(); ++i) {
    auto value = op.result(i);
    std::string name = CompatibleInfo::OutputName(value);
    names.push_back(std::move(name));
  }
  return names;
}

}  // namespace newir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
