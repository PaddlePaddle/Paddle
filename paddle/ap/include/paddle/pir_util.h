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

namespace ap::paddle {

struct FreeVarHelper {
  void GetUsedExternalValueImpl(
      std::unordered_set<pir::Value>& defined_values,  // NOLINT
      std::vector<pir::Value>& used_values,            // NOLINT
      const pir::Operation& op) {
    for (size_t index = 0; index < op.num_operands(); ++index) {
      pir::Value value = op.operand_source(index);
      if (defined_values.find(value) == defined_values.end()) {
        used_values.push_back(value);
        defined_values.insert(value);
      }
    }
    for (auto& region : op) {
      for (auto& block : region) {
        for (auto value : block.args()) {
          defined_values.insert(value);
        }
        for (const auto& [_, value] : block.kwargs()) {
          defined_values.insert(value);
        }
      }
      for (auto& block : region) {
        for (auto& inner_op : block) {
          GetUsedExternalValueImpl(defined_values, used_values, inner_op);
        }
      }
    }
    for (size_t index = 0; index < op.num_results(); ++index) {
      defined_values.insert(op.result(index));
    }
  }

  std::vector<pir::Value> GetUsedExternalValue(const pir::Operation& op) {
    std::unordered_set<pir::Value> defined_values{nullptr};
    std::vector<pir::Value> used_values;
    GetUsedExternalValueImpl(defined_values, used_values, op);
    return used_values;
  }
};

inline std::vector<pir::Value> GetUsedExternalValue(const pir::Operation& op) {
  return FreeVarHelper{}.GetUsedExternalValue(op);
}

}  // namespace ap::paddle
