/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

struct Operation {
  Operation() = default;
  Operation(int t, int n, std::string o, std::vector<std::string> e,
            std::vector<std::string> i_n, std::vector<std::string> o_n)
      : type(t),
        num_operands(n),
        op_type(o),
        exprs(e),
        input_names(i_n),
        output_names(o_n) {}

  bool IsGradOp() {
    std::string suffix = "_grad";
    size_t pos = op_type.rfind(suffix);
    return pos != std::string::npos &&
           pos == (op_type.length() - suffix.length());
  }

  bool IsValid() {
    if (!IsGradOp() && exprs.size() != 1U) {
      // When it is a forward operation, it should hold only one expression (for
      // only one output).
      return false;
    }
    if (IsGradOp() && exprs.size() != static_cast<size_t>(num_operands)) {
      // When it is a backward opertion, it should hold a expression for each
      // operand.
      return false;
    }
    return true;
  }

  int type;
  int num_operands;
  std::string op_type;
  std::vector<std::string> exprs;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
};

class OperationMap {
 public:
  OperationMap();

  static OperationMap& Instance() {
    PADDLE_ENFORCE_NOT_NULL(
        map, platform::errors::PreconditionNotMet(
                 "Please initialize OperationMap first, by calling "
                 "framework::fusion_group::OperationMap::Init()!"));
    return *map;
  }

  static OperationMap& Init() {
    if (map == nullptr) {
      map = new OperationMap();
    }
    return *map;
  }

  std::unordered_set<std::string> Find(int type);

  bool Has(std::string op_type) {
    return operations_.find(op_type) != operations_.end();
  }

  Operation& Get(std::string op_type) {
    auto iter = operations_.find(op_type);
    PADDLE_ENFORCE_NE(iter, operations_.end(),
                      platform::errors::Unimplemented(
                          "Operation %s is not supported yet.", op_type));
    return iter->second;
  }

 private:
  void Insert(int type, int num_operands, std::string op_type, std::string expr,
              std::vector<std::string> grad_exprs,
              std::vector<std::string> input_names,
              std::vector<std::string> output_names);

  void InsertUnaryElementwiseOperations();
  void InsertBinaryElementwiseOperations();
  void InsertMultivariateElementwiseOperations();

 private:
  static OperationMap* map;
  std::unordered_map<std::string, Operation> operations_;
  DISABLE_COPY_AND_ASSIGN(OperationMap);
};

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
