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
  int type;
  int num_operands;
  std::string op_type;
  std::string expr;
  std::vector<std::string> grad_exprs;
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
