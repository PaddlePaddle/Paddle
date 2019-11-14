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
  Operation() {}
  Operation(int t, int n, std::string o, std::vector<std::string> e)
      : type(t), num_operands(n), op_type(o), exprs(e) {}

  bool IsGradOp() {
    std::string suffix = "_grad";
    return op_type.rfind(suffix) == (op_type.length() - suffix.length());
  }

  bool IsValid() {
    if (!IsGradOp() && exprs.size() != 1U) {
      return false;
    }
    if (IsGradOp() && exprs.size() != static_cast<size_t>(num_operands)) {
      return false;
    }
    return true;
  }

  int type;
  int num_operands;
  std::string op_type;
  std::vector<std::string> exprs;
};

class OperationMap {
 public:
  OperationMap();

  static OperationMap& Instance() {
    PADDLE_ENFORCE_NOT_NULL(map, "Need to initialize OperationMap first!");
    return *map;
  }

  static OperationMap& Init() {
    if (map == nullptr) {
      map = new OperationMap();
    }
    return *map;
  }

  std::unordered_set<std::string> Find(int type, int num_operands = -1);

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
              std::vector<std::string> grad_exprs);

  void InsertUnaryElementwiseOperations();
  void InsertBinaryElementwiseOperations();

 private:
  static OperationMap* map;
  std::unordered_map<std::string, Operation> operations_;
  DISABLE_COPY_AND_ASSIGN(OperationMap);
};

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
